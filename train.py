"""
Distorted Visual Sequence Pattern Recognition — CRNN + CTC
=============================================================
Architecture : ResNet-style CNN → BiLSTM (x2) → CTC Loss
Decode       : CTC Beam Search (width=10) + TTA (5 passes)
Metric       : Character Error Rate (CER)

Usage:
    python train.py

Requirements:
    pip install torch torchvision pillow pandas matplotlib

Expected dataset layout:
    cig_ps/
        train_images/   # train-0.png ... train-19999.png
        test_images/    # test-0.png  ... test-4999.png
        train-labels.csv
"""

import os, sys, random, time, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 42
DATA_DIR      = Path('cig_ps')
TRAIN_DIR     = DATA_DIR / 'train_images'
TEST_DIR      = DATA_DIR / 'test_images'
LABEL_CSV     = DATA_DIR / 'train-labels.csv'
MODEL_DIR     = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH    = MODEL_DIR / 'crnn_best.pt'

IMG_H         = 48
IMG_W         = 160

EPOCHS        = 50
BATCH_SIZE    = 128
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 5.0
VAL_SIZE      = 2000
PATIENCE      = 12

BEAM_WIDTH    = 10
TTA_N         = 5

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREP
# ─────────────────────────────────────────────────────────────────────────────
df_full  = pd.read_csv(LABEL_CSV)
df_full  = df_full[df_full['text'].str.match(r'^[A-Z0-9]{6}$')].reset_index(drop=True)

CHARSET     = sorted(set(''.join(df_full['text'].tolist())))
BLANK_IDX   = 0
char2idx    = {c: i+1 for i, c in enumerate(CHARSET)}
idx2char    = {i+1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1

print(f'Train samples : {len(df_full):,}')
print(f'Charset ({len(CHARSET)}): {"".join(CHARSET)}')
print(f'NUM_CLASSES   : {NUM_CLASSES}')


def encode_label(text):
    return [char2idx[c] for c in text]


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class CaptchaAugment:
    def __call__(self, img):
        if random.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        if random.random() < 0.45:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.45))
        if random.random() < 0.45:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.55))
        if random.random() < 0.25:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.5))
        if random.random() < 0.30:
            shift = random.randint(-8, 8)
            img = TF.affine(img, angle=0, translate=[shift, 0], scale=1.0, shear=0)
        if random.random() < 0.30:
            angle = random.uniform(-3, 3)
            img = TF.rotate(img, angle, fill=200)
        if random.random() < 0.25:
            w, h = img.size
            img = img.resize((int(w * random.uniform(0.9, 1.1)), h), Image.BILINEAR)
        return img


def build_transform(augment=False):
    ops = []
    if augment:
        ops.append(CaptchaAugment())
    ops += [
        transforms.Grayscale(),
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    if augment:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0))
    return transforms.Compose(ops)


TRAIN_TRANSFORM = build_transform(augment=True)
EVAL_TRANSFORM  = build_transform(augment=False)


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class CaptchaDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.img_dir / row['image']).convert('RGB')
        img   = self.transform(img)
        label = torch.tensor(encode_label(row['text']), dtype=torch.long)
        return img, label, len(row['text'])


class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.files     = sorted(
            os.listdir(img_dir),
            key=lambda x: int(x.split('-')[1].split('.')[0])
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(self.img_dir / fname).convert('RGB')
        return self.transform(img), fname


def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.tensor(lengths, dtype=torch.long)


df_val   = df_full.sample(VAL_SIZE, random_state=SEED)
df_train = df_full.drop(df_val.index).reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)

train_ds = CaptchaDataset(df_train, TRAIN_DIR, TRAIN_TRANSFORM)
val_ds   = CaptchaDataset(df_val,   TRAIN_DIR, EVAL_TRANSFORM)
test_ds  = TestDataset(TEST_DIR, EVAL_TRANSFORM)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f'Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(test_ds):,}')


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetCRNN(nn.Module):
    """
    ResNet-style CNN backbone → BiLSTM × 2 → Linear head → CTC

    Input  : B × 1 × 48 × 160
    CNN out: B × 512 × 1 × 40   (40 time steps)
    RNN    : BiLSTM hidden=256, 2 layers
    Output : 40 × B × NUM_CLASSES  (CTC format: T × N × C)
    """
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 48×160 → 24×80
        )
        self.stage1 = nn.Sequential(
            ResBlock(64, 128, stride=2),                 # 24×80  → 12×40
        )
        self.stage2 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.MaxPool2d((2,1), (2,1)),                  # 12×40  → 6×40
        )
        self.stage3 = nn.Sequential(
            ResBlock(256, 512),
            ResBlock(512, 512),
            nn.MaxPool2d((2,1), (2,1)),                  # 6×40   → 3×40
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))      # 3×40   → 1×40

        self.rnn = nn.LSTM(
            input_size=512, hidden_size=rnn_hidden, num_layers=rnn_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(rnn_hidden * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        f = self.dropout(f.squeeze(2).permute(0, 2, 1))  # B × W' × 512
        out, _ = self.rnn(f)
        out = self.head(self.dropout(out))                # B × W' × C
        return out.permute(1, 0, 2)                       # T × B × C


model = ResNetCRNN(NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {total_params:,}')

# Verify shapes
with torch.no_grad():
    _out = model(torch.zeros(2, 1, IMG_H, IMG_W).to(DEVICE))
    print(f'Output shape: {_out.shape}  (T × B × C)')


# ─────────────────────────────────────────────────────────────────────────────
# LOSS / OPTIMIZER / SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)

no_decay = {'bias', 'bn', 'BatchNorm'}
optimizer = torch.optim.AdamW([
    {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
], lr=LR)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS,
    pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=1e4,
)

use_amp = (DEVICE.type == 'cuda')
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
print(f'AMP: {use_amp} | Scheduler: OneCycleLR')


# ─────────────────────────────────────────────────────────────────────────────
# DECODING
# ─────────────────────────────────────────────────────────────────────────────
def greedy_decode(log_probs_BTC):
    indices = log_probs_BTC.argmax(dim=2).cpu().numpy()
    results = []
    for row in indices:
        chars, prev = [], None
        for idx in row:
            if idx != prev:
                if idx != BLANK_IDX:
                    chars.append(idx2char.get(int(idx), ''))
                prev = idx
        results.append(''.join(chars))
    return results


def beam_search_single(log_probs_TC, beam_width=10):
    probs = np.exp(log_probs_TC)
    beam  = {(): (1.0, 0.0)}

    for t in range(len(probs)):
        new_beam = defaultdict(lambda: (0.0, 0.0))
        top = sorted(beam.items(), key=lambda x: x[1][0]+x[1][1], reverse=True)[:beam_width]

        for prefix, (p_b, p_nb) in top:
            p_tot = p_b + p_nb
            pb, pnb = new_beam[prefix]
            new_beam[prefix] = (pb + p_tot * probs[t, BLANK_IDX], pnb)

            for c in range(1, probs.shape[1]):
                new_pfx = prefix + (c,)
                pb, pnb = new_beam[new_pfx]
                if prefix and prefix[-1] == c:
                    new_beam[new_pfx] = (pb, pnb + p_b * probs[t, c])
                else:
                    new_beam[new_pfx] = (pb, pnb + p_tot * probs[t, c])

        beam = dict(new_beam)

    best = max(beam.items(), key=lambda x: x[1][0]+x[1][1])
    return ''.join(idx2char.get(i, '') for i in best[0])


def beam_decode_batch(log_probs_BTC, beam_width=10):
    arr = log_probs_BTC.cpu().numpy()
    return [beam_search_single(arr[b], beam_width) for b in range(arr.shape[0])]


def postprocess(pred, n=6):
    if len(pred) == n: return pred
    if len(pred) > n:  return pred[:n]
    return pred + (pred[-1] if pred else CHARSET[0]) * (n - len(pred))


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def levenshtein(s1, s2):
    if len(s1) < len(s2): s1, s2 = s2, s1
    if not s2: return len(s1)
    prev = list(range(len(s2)+1))
    for i, c1 in enumerate(s1):
        curr = [i+1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]

def compute_cer(preds, targets):
    return sum(levenshtein(p,t) for p,t in zip(preds,targets)) / max(sum(len(t) for t in targets),1)

def compute_acc(preds, targets):
    return sum(p==t for p,t in zip(preds,targets)) / len(targets)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(train=True):
    loader = train_loader if train else val_loader
    model.train(train)
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels, lengths in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            T, B, C = logits.shape
            il   = torch.full((B,), T, dtype=torch.long, device=DEVICE)
            loss = ctc_loss_fn(logits.log_softmax(2), labels, il, lengths.to(DEVICE))

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            lp = logits.log_softmax(2).permute(1, 0, 2)
            batch_preds = greedy_decode(lp)
            all_preds.extend([postprocess(p) for p in batch_preds])
            offset, lbl_cpu = 0, labels.cpu().tolist()
            for length in lengths.tolist():
                all_targets.append(''.join(idx2char[i] for i in lbl_cpu[offset:offset+length]))
                offset += length

    return total_loss / len(loader), compute_cer(all_preds, all_targets), compute_acc(all_preds, all_targets)


history = {'epoch':[], 'tr_loss':[], 'vl_loss':[], 'vl_cer':[], 'vl_acc':[]}
best_cer, patience_ctr = float('inf'), 0

print(f'\n{"="*70}')
print(f'Training: {EPOCHS} epochs | Batch={BATCH_SIZE} | LR={LR} | Patience={PATIENCE}')
print(f'{"="*70}')

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_cer, tr_acc = run_epoch(train=True)
    with torch.no_grad():
        vl_loss, vl_cer, vl_acc = run_epoch(train=False)
    elapsed = time.time() - t0
    lr_now  = optimizer.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['tr_loss'].append(tr_loss); history['vl_loss'].append(vl_loss)
    history['vl_cer'].append(vl_cer);   history['vl_acc'].append(vl_acc)

    tag = ''
    if vl_cer < best_cer:
        best_cer = vl_cer
        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(),
            'best_cer': best_cer, 'charset': CHARSET,
            'config': {'IMG_H': IMG_H, 'IMG_W': IMG_W, 'NUM_CLASSES': NUM_CLASSES}
        }, MODEL_PATH)
        patience_ctr = 0
        tag = '  ← BEST'
    else:
        patience_ctr += 1

    print(f'[{epoch:02d}/{EPOCHS}] '
          f'tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | '
          f'vl_loss={vl_loss:.4f} vl_CER={vl_cer:.4f} vl_acc={vl_acc:.3f} | '
          f'lr={lr_now:.2e} | {elapsed:.1f}s{tag}')

    if patience_ctr >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}.')
        break

print(f'\nBest val CER (greedy): {best_cer:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['epoch'], history['tr_loss'], label='Train')
axes[0].plot(history['epoch'], history['vl_loss'], label='Val')
axes[0].set_title('CTC Loss'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history['epoch'], history['vl_cer'], color='tomato')
axes[1].axhline(best_cer, ls='--', color='red', label=f'Best={best_cer:.4f}')
axes[1].set_title('Val CER'); axes[1].legend(); axes[1].grid(True)

axes[2].plot(history['epoch'], history['vl_acc'], color='steelblue')
axes[2].set_title('Val Exact Acc'); axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=120, bbox_inches='tight')
print('Training curves saved: training_curves.png')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD BEST MODEL & BEAM-SEARCH EVAL ON VAL
# ─────────────────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"\nLoaded best model (epoch {ckpt['epoch']}, CER={ckpt['best_cer']:.4f})")

print('Running beam search on validation set...')
val_preds, val_targets = [], []
with torch.no_grad():
    for imgs, labels, lengths in val_loader:
        imgs      = imgs.to(DEVICE, non_blocking=True)
        logits    = model(imgs)
        lp        = logits.log_softmax(2).permute(1, 0, 2)
        batch_p   = beam_decode_batch(lp.cpu(), BEAM_WIDTH)
        val_preds.extend([postprocess(p) for p in batch_p])
        offset, lbl_cpu = 0, labels.cpu().tolist()
        for length in lengths.tolist():
            val_targets.append(''.join(idx2char[i] for i in lbl_cpu[offset:offset+length]))
            offset += length

final_cer = compute_cer(val_preds, val_targets)
final_acc = compute_acc(val_preds, val_targets)
print(f'Val CER  (beam, w={BEAM_WIDTH}): {final_cer:.4f}')
print(f'Val Acc  (beam, w={BEAM_WIDTH}): {final_acc:.4f}  ({final_acc*100:.2f}%)')


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE WITH TTA + BEAM SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# TTA ops applied in tensor space (after normalization)
tta_ops = [
    lambda x: x,                                               # original
    lambda x: torch.clamp(x * 0.85 + 0.03 * torch.randn_like(x), -1, 1),
    lambda x: torch.clamp(x * 1.15, -1, 1),
    lambda x: torch.clamp(x * 0.90, -1, 1),
    lambda x: x + 0.04 * torch.randn_like(x),
]
tta_ops = tta_ops[:TTA_N]

print(f'\nRunning TTA inference ({TTA_N} passes) + Beam Search (w={BEAM_WIDTH})...')
test_preds, test_fnames = [], []
model.eval()
t0 = time.time()

with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        sum_lp = None
        for op in tta_ops:
            logits = model(op(imgs))
            lp     = logits.log_softmax(2).permute(1, 0, 2)
            sum_lp = lp if sum_lp is None else sum_lp + lp
        avg_lp = sum_lp / len(tta_ops)
        batch_p = beam_decode_batch(avg_lp.cpu(), BEAM_WIDTH)
        test_preds.extend([postprocess(p) for p in batch_p])
        test_fnames.extend(list(fnames))

elapsed = time.time() - t0
print(f'Inference done: {len(test_preds)} samples in {elapsed:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION CSV
# ─────────────────────────────────────────────────────────────────────────────
submission = pd.DataFrame({'image': test_fnames, 'prediction': test_preds})

# Assertions
assert len(submission) == len(test_ds)
assert (submission['prediction'].str.len() == 6).all()

OUT = 'submission_Atharva_CRNN.csv'
submission.to_csv(OUT, index=False)

print(f'\nSubmission saved: {OUT}')
print(submission.head(10).to_string(index=False))
print(f'\nPred length dist: {submission["prediction"].str.len().value_counts().to_dict()}')

print('\n' + '='*65)
print('  FINAL RESULTS')
print('='*65)
print(f'  Val CER (greedy, best epoch) : {best_cer:.4f}')
print(f'  Val CER (beam, w={BEAM_WIDTH})       : {final_cer:.4f}')
print(f'  Val Exact Acc                : {final_acc:.4f}  ({final_acc*100:.2f}%)')
print(f'  Submission file              : {OUT}')
print('='*65)
