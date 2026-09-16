# Local Python conversion of the original Colab notebook.
import os

# Â§3 â€” Imports, config & reproducibility
import os, cv2, random, time, warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import timm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT       = os.getenv("MASTITIS_DATA_ROOT", os.path.join(os.path.dirname(__file__), "data"))
CLASS_NAMES = ["Healthy", "Mastitic"]                    # index 0, 1  (positive = Mastitis = 1)
POS_IDX    = 1

IMG_SIZE   = 224          # 224 is required by ViT-B/16 and Swin-Tiny
BATCH_SIZE = 16           # â†“ to 8 if you hit CUDA OOM;  â†‘ to 32 on A100/L4
NUM_WORKERS = 2           # DataLoader workers; set 0 if you hit 'used all RAM'
EPOCHS     = 20
LR         = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE   = 6            # early-stopping patience (epochs w/o val-AUC gain)
USE_CLAHE  = True
SEED       = 42

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test  (cow-level)

OUT_DIR = os.path.join(ROOT, "DL_results")
os.makedirs(OUT_DIR, exist_ok=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

device  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (device == "cuda")
print(f"Device: {device} | timm {timm.__version__} | torch {torch.__version__}")
if device == "cpu":
    print("âš ï¸  No GPU detected â€” training will be slow. Enable GPU in Runtime settings.")

import imagehash
from PIL import Image
import os
from collections import defaultdict

# Ø§Ø­Ø³Ø¨ÙŠ "Ø¨ØµÙ…Ø©" (perceptual hash) Ù„ÙƒÙ„ ØµÙˆØ±Ø© ÙÙŠ Ø§Ù„Ù…Ø¬Ù„Ø¯ÙŠÙ† Ø§Ù„Ø£ØµÙ„ÙŠÙŠÙ†
hashes = defaultdict(list)
for cls in ["Healthy", "Mastitic"]:
    d = os.path.join(ROOT, cls)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".bmp",".png",".jpg",".jpeg")):
            try:
                h = imagehash.phash(Image.open(os.path.join(d, f)).convert("RGB"))
                hashes[str(h)].append((cls, f))
            except Exception as e:
                print("skip", f, e)

# Ø¬Ù…Ù‘Ø¹ÙŠ Ø§Ù„ØµÙˆØ± Ø§Ù„Ù…ØªØ´Ø§Ø¨Ù‡Ø© Ø¬Ø¯Ù‹Ø§ (ÙØ±Ù‚ ØµÙØ± Ø£Ùˆ Ø­Ø¯ Ø£Ø¯Ù†Ù‰) ÙƒÙ€"Ù†ÙØ³ Ø§Ù„ØµÙˆØ±Ø© Ø§Ù„Ø£ØµÙ„ÙŠØ©"
clusters = {h: v for h, v in hashes.items() if len(v) > 1}
print(f"Ø¥Ø¬Ù…Ø§Ù„ÙŠ Ø§Ù„ØµÙˆØ±: {sum(len(v) for v in hashes.values())}")
print(f"Ø¹Ø¯Ø¯ Ø§Ù„Ù€ hash Ø§Ù„ÙØ±ÙŠØ¯Ø©: {len(hashes)}")
print(f"Ø¹Ø¯Ø¯ Ø§Ù„ÙƒÙ„Ø§Ø³ØªØ±Ø² (ØµÙˆØ± Ø¨ØªØªÙƒØ±Ø±): {len(clusters)}")
# Ù„Ùˆ Ø¹Ø¯Ø¯ Ø§Ù„Ù€ hash Ø§Ù„ÙØ±ÙŠØ¯Ø© Ù‚Ø±ÙŠØ¨ Ù…Ù† 976 (Ø£Ùˆ 488)ØŒ ÙŠØ¨Ù‚Ù‰ ÙØ¹Ù„Ø§Ù‹ ÙÙŠÙ‡ ØªÙƒØ±Ø§Ø±/augmentation Ù‚Ø¨Ù„ Ø§Ù„Ø­ÙØ¸

# Â§4 â€” Cow-level stratified split  (Healthy/ + Mastitis/  â†’  train/val/test on disk)
# The same cow (identified from the filename) is kept entirely within ONE split,
# so no cow leaks across train/test. This prevents pseudoreplication.
import shutil

VALID_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

def cow_id_from_name(fname):
    """Cow ID = filename without extension / trailing dots (e.g. '57..bmp' -> '57')."""
    base = os.path.splitext(fname)[0].rstrip(".").strip()
    return base

def list_class_files(class_name):
    d = os.path.join(ROOT, class_name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing class folder: {d}")
    return [f for f in os.listdir(d) if f.lower().endswith(VALID_EXT)]

def split_cow_groups(files, ratios, seed=SEED):
    """Split UNIQUE cow-groups (not images) into train/val/test."""
    groups = sorted({cow_id_from_name(f) for f in files})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    # guarantee non-empty val/test when enough cows exist
    if n >= 3:
        n_va = max(1, n_va)
        n_tr = min(n_tr, n - 2)
    train_g = set(groups[:n_tr])
    val_g   = set(groups[n_tr:n_tr + n_va])
    test_g  = set(groups[n_tr + n_va:])
    return train_g, val_g, test_g

def materialize_split():
    # clean existing split dirs so re-runs are reproducible
    for sub in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            p = os.path.join(ROOT, sub, cls)
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p, exist_ok=True)

    summary = {}
    for cls in CLASS_NAMES:
        files = list_class_files(cls)
        tr_g, va_g, te_g = split_cow_groups(files, SPLIT_RATIOS)
        counts = {"train": 0, "val": 0, "test": 0}
        for f in files:
            g = cow_id_from_name(f)
            sub = "train" if g in tr_g else ("val" if g in va_g else "test")
            shutil.copy2(os.path.join(ROOT, cls, f),
                         os.path.join(ROOT, sub, cls, f))
            counts[sub] += 1
        summary[cls] = {
            "images": counts,
            "cows": {"train": len(tr_g), "val": len(va_g), "test": len(te_g)}
        }
    return summary

print("Splitting at cow level (this copies files into train/val/test on your Drive)...")
summary = materialize_split()

print("\n%-10s | %-22s | %-22s" % ("Class", "images (tr/val/test)", "cows (tr/val/test)"))
print("-" * 62)
for cls, s in summary.items():
    im, cw = s["images"], s["cows"]
    print("%-10s | %-22s | %-22s" % (
        cls,
        f'{im["train"]}/{im["val"]}/{im["test"]}',
        f'{cw["train"]}/{cw["val"]}/{cw["test"]}'))
print("\nâœ… Split written to", os.path.join(ROOT, "{train,val,test}"))

# Â§5 â€” Preprocessing & augmentation transforms (CLAHE + ImageNet normalization)
# RAM-safe order: RESIZE FIRST so CLAHE runs on a small image, not the full-size BMP.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CLAHE:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel (LAB)."""
    def __init__(self, clip=2.0, grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    def __call__(self, img):
        arr = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)

_clahe = [CLAHE()] if USE_CLAHE else []

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),          # shrink first â†’ CLAHE & aug use far less RAM
    *_clahe,
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    *_clahe,
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
print("Transforms ready | CLAHE:", USE_CLAHE, "| resize-before-CLAHE (RAM-safe)")

# Â§6 â€” Datasets, DataLoaders & class weights
train_ds = datasets.ImageFolder(os.path.join(ROOT, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(ROOT, "val"),   transform=eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(ROOT, "test"),  transform=eval_tf)

# Ensure label order matches CLASS_NAMES (ImageFolder is alphabetical: Healthy=0, Mastitis=1)
assert train_ds.classes == CLASS_NAMES, f"Unexpected class order: {train_ds.classes}"

def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=(device == "cuda"), drop_last=False)

train_loader = make_loader(train_ds, True)
val_loader   = make_loader(val_ds,   False)
test_loader  = make_loader(test_ds,  False)

# class-weighted loss to counter imbalance:  w_c = N / (K * count_c)
cnt = Counter(train_ds.targets)
N, K = len(train_ds.targets), len(CLASS_NAMES)
class_weights = torch.tensor([N / (K * cnt[i]) for i in range(K)], dtype=torch.float32)

print("Train:", Counter(train_ds.targets))
print("Val:  ", Counter(val_ds.targets))
print("Test: ", Counter(test_ds.targets))
print("Class weights (Healthy, Mastitis):", class_weights.tolist())

# Â§7 â€” Model factory (all 7 architectures via timm)
MODEL_ZOO = {
    "ResNet-50":       "resnet50",
    "DenseNet-121":    "densenet121",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2":     "mobilenetv2_100",
    "Inception-V3":    "inception_v3",
    "ViT-B/16":        "vit_base_patch16_224",
    "Swin-Tiny":       "swin_tiny_patch4_window7_224",
}

def build_model(timm_name):
    """Pretrained ImageNet backbone with a fresh 2-class head."""
    model = timm.create_model(timm_name, pretrained=True, num_classes=len(CLASS_NAMES))
    return model.to(device)

print("Registered models:")
for k, v in MODEL_ZOO.items():
    print(f"  â€¢ {k:<16} â†’ {v}")

# Â§8 â€” Training & evaluation engine (AMP, cosine LR, early stopping on val-AUC)
@torch.no_grad()
def predict(model, loader):
    """Return (y_true, y_prob_positive)."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = model(x)
        prob = torch.softmax(out.float(), dim=1)[:, POS_IDX]
        ps.append(prob.cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy":    accuracy_score(y_true, y_pred),
        "AUC":         roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,   # recall for Mastitis
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
    }

def train_model(model, name, tloader=None, vloader=None, epochs=EPOCHS):
    tloader = train_loader if tloader is None else tloader
    vloader = val_loader   if vloader is None else vloader
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_auc, best_state, wait = -1.0, None, 0
    history = {"train_loss": [], "val_auc": []}

    for ep in range(1, epochs + 1):
        model.train(); run_loss, seen = 0.0, 0
        pbar = tqdm(tloader, desc=f"  {name} ep{ep:02d}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss += loss.item() * x.size(0); seen += x.size(0)
            pbar.set_postfix(loss=f"{run_loss/seen:.4f}")
        scheduler.step()

        yv, pv = predict(model, vloader)
        vauc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else 0.5
        history["train_loss"].append(run_loss / max(seen, 1))
        history["val_auc"].append(vauc)
        print(f"  ep{ep:02d} | train_loss {run_loss/seen:.4f} | val_AUC {vauc:.4f}")

        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  â¹ early stop @ ep{ep} (best val_AUC {best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_auc

import os, glob
print("Part 2 (CV) cached:")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "cv_result_*.pkl"))):
    print("  âœ”", os.path.basename(f).replace("cv_result_", "").replace(".pkl", ""))
print("\nPart 1 cached:")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "result_*.pkl"))):
    print("  âœ”", os.path.basename(f).replace("result_", "").replace(".pkl", ""))

import gc, pickle

def result_path(timm_name):
    return os.path.join(OUT_DIR, f"result_{timm_name}.pkl")

# ðŸ’¡ If RAM is tight, train in CHUNKS: set MODELS_THIS_RUN to 2â€“3 names,
#    run this cell, then Runtimeâ†’Restart, re-run Â§1â€“Â§8, change the list, run again.
#    Finished models are cached to Drive and skipped automatically.
MODELS_THIS_RUN = list(MODEL_ZOO)     # e.g. ["ResNet-50", "DenseNet-121"]

results = {}
# reload anything already computed in a previous (possibly crashed) run
for name, timm_name in MODEL_ZOO.items():
    p = result_path(timm_name)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                results[name] = pickle.load(f)
            print(f"â†©ï¸Ž  loaded cached result: {name}")
        except OSError as e:
            print(f"âš ï¸ Error loading cached result for {name} from {p}: {e}")
            print(f"   This model will be re-trained if it's in MODELS_THIS_RUN.")
        except Exception as e:
            print(f"âš ï¸ Unexpected error loading cached result for {name} from {p}: {e}")

for name in MODELS_THIS_RUN:
    if name in results:
        continue                        # already trained -> skip (resume)
    timm_name = MODEL_ZOO[name]
    print("\n" + "=" * 68)
    print(f"ðŸš€ {name}  ({timm_name})")
    print("=" * 68)
    set_seed()
    model = build_model(timm_name)

    t0 = time.time()
    model, hist, best_val_auc = train_model(model, name)
    train_min = (time.time() - t0) / 60

    y_true, y_prob = predict(model, test_loader)
    m = compute_metrics(y_true, y_prob)
    results[name] = {
        "metrics": m, "y_true": y_true, "y_prob": y_prob,
        "history": hist, "best_val_auc": best_val_auc, "train_min": train_min,
    }
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{timm_name}.pt"))
    with open(result_path(timm_name), "wb") as f:      # checkpoint metrics/predictions
        pickle.dump(results[name], f)

    # â”€â”€ free memory before the next model â”€â”€
    del model, y_true, y_prob
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  âœ” TEST  Acc {m['Accuracy']:.3f} | AUC {m['AUC']:.3f} | "
          f"Sens {m['Sensitivity']:.3f} | Spec {m['Specificity']:.3f} | "
          f"F1 {m['F1']:.3f}  ({train_min:.1f} min)")

print(f"\nâœ… {len(results)}/{len(MODEL_ZOO)} models available (cached in {OUT_DIR}).")

rows = []
for name, r in results.items():
    row = {"Model": name, **{k: round(v, 4) for k, v in r["metrics"].items()}}
    row["Train_min"] = round(r["train_min"], 1)
    rows.append(row)

df = (pd.DataFrame(rows)
      .sort_values("AUC", ascending=False)
      .reset_index(drop=True))
df.index += 1
csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
try:
    df.to_csv(csv_path)
    print("Saved:", csv_path)
except OSError as e:
    print(f"âš ï¸ Error saving model comparison CSV to {csv_path}: {e}")
    print("   Continuing without saving this file.")

best_name = df.iloc[0]["Model"]
print(f"ðŸ† Best model by AUC: {best_name}\n")
df

# Â§11 â€” Grouped bar chart: Accuracy / AUC / Sensitivity / Specificity / F1
metrics_to_plot = ["Accuracy", "AUC", "Sensitivity", "Specificity", "F1"]
order = df["Model"].tolist()
data = {m: [results[n]["metrics"][m] for n in order] for m in metrics_to_plot}

x = np.arange(len(order)); w = 0.16
plt.figure(figsize=(13, 6))
for i, m in enumerate(metrics_to_plot):
    plt.bar(x + (i - 2) * w, data[m], w, label=m)
plt.xticks(x, order, rotation=20, ha="right")
plt.ylim(0, 1.05); plt.ylabel("Score")
plt.title("Deep Learning & Vision Transformers â€” Test Performance")
plt.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28))
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparison_bars.png"), dpi=200, bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import os

# Â§12 â€” ROC curves (all 7 models overlaid)
plt.figure(figsize=(8, 7))
for name in order:
    r = results[name]
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curves â€” Mastitis Detection (Test Set)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3); plt.tight_layout()

try:
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=200, bbox_inches="tight")
    print("Saved ROC curves to:", os.path.join(OUT_DIR, "roc_curves.png"))
except OSError as e:
    print(f"âš ï¸ Error saving ROC curves to {os.path.join(OUT_DIR, 'roc_curves.png')}: {e}")
    print("   Continuing without saving this file.")
plt.show()


n = len(order); cols = 4; rows_ = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows_, cols, figsize=(4 * cols, 3.6 * rows_))
axes = np.array(axes).reshape(-1)
for ax, name in zip(axes, order):
    r = results[name]
    y_pred = (r["y_prob"] >= 0.5).astype(int)
    cm = confusion_matrix(r["y_true"], y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{name}\nAcc={r['metrics']['Accuracy']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for ax in axes[n:]:
    ax.axis("off")
plt.tight_layout()
try:
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=200, bbox_inches="tight")
    print("Saved confusion matrices to:", os.path.join(OUT_DIR, "confusion_matrices.png"))
except OSError as e:
    print(f"âš ï¸ Error saving confusion matrices to {os.path.join(OUT_DIR, 'confusion_matrices.png')}: {e}")
    print("   Continuing without saving this file.")
plt.show()

# Â§14 â€” Bootstrap 95% confidence intervals (best model)
def bootstrap_ci(y_true, y_prob, n_boot=1000, thr=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = ["Accuracy", "AUC", "Sensitivity", "Specificity", "F1"]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, pt = y_true[idx], y_prob[idx]
        if len(set(yt)) < 2:
            continue
        m = compute_metrics(yt, pt, thr)
        for k in keys:
            store[k].append(m[k])
    print(f"{'Metric':<13}{'Mean':>8}   95% CI")
    print("-" * 40)
    for k in keys:
        a = np.array(store[k]); lo, hi = np.percentile(a, [2.5, 97.5])
        print(f"{k:<13}{a.mean():>8.4f}   ({lo:.4f} â€“ {hi:.4f})")

print(f"Bootstrap CIs â€” {best_name}\n")
bootstrap_ci(results[best_name]["y_true"], results[best_name]["y_prob"])

# Â§15 â€” Training-loss & validation-AUC curves
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
for name in order:
    h = results[name]["history"]
    a1.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=name)
    a2.plot(range(1, len(h["val_auc"]) + 1),    h["val_auc"],    label=name)
a1.set_title("Training Loss"); a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.grid(alpha=0.3)
a2.set_title("Validation AUC"); a2.set_xlabel("Epoch"); a2.set_ylabel("AUC"); a2.grid(alpha=0.3)
a2.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=200, bbox_inches="tight")
plt.show()

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def last_conv(model):
    conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    return conv

def denorm(t):
    x = t.clone().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(x, 0, 1)

cam_timm = MODEL_ZOO[best_name]
is_transformer = cam_timm.startswith(("vit", "swin"))

if is_transformer:
    print(f"â„¹ï¸  Best model ({best_name}) is a transformer. Grad-CAM needs a reshape_transform "
          "for attention maps â€” shown here for a CNN instead for a clean heatmap.")
    cam_name = next(n for n in order if not MODEL_ZOO[n].startswith(("vit", "swin")))
    cam_timm = MODEL_ZOO[cam_name]
else:
    cam_name = best_name

cam_model = None # Initialize to None

try:
    # reload chosen CNN
    cam_model = build_model(cam_timm)
    model_path = os.path.join(OUT_DIR, f"{cam_timm}.pt")
    cam_model.load_state_dict(torch.load(model_path, map_location=device))
    cam_model.eval()

    # grab a few test images (mix of both classes)
    samples, seen_lbl = [], set()
    for path, lbl in test_ds.samples:
        if lbl not in seen_lbl or len(samples) < 6:
            samples.append((path, lbl)); seen_lbl.add(lbl)
        if len(samples) >= 6:
            break

    cam = GradCAM(model=cam_model, target_layers=[last_conv(cam_model)])
    plt.figure(figsize=(15, 5))
    for i, (path, lbl) in enumerate(samples[:6]):
        img = Image.open(path).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(cam_model(x).float(), 1)[0, POS_IDX].item()
        grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(POS_IDX)])[0]
        vis = show_cam_on_image(denorm(x[0]), grayscale, use_rgb=True)
        plt.subplot(2, 3, i + 1)
        plt.imshow(vis); plt.axis("off")
        plt.title(f"True: {CLASS_NAMES[lbl]} | P(mast)={prob:.2f}", fontsize=9)
    plt.suptitle(f"Grad-CAM â€” {cam_name}", y=1.02)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(OUT_DIR, "gradcam.png"), dpi=200, bbox_inches="tight")
        print(f"Saved Grad-CAM to: {os.path.join(OUT_DIR, 'gradcam.png')}")
    except OSError as e:
        print(f"âš ï¸ Error saving Grad-CAM image to {os.path.join(OUT_DIR, 'gradcam.png')}: {e}")
        print("   Continuing without saving this file.")
    plt.show()

except OSError as e:
    print(f"âš ï¸ Error loading model for Grad-CAM from {os.path.join(OUT_DIR, f'{cam_timm}.pt')}: {e}")
    print("   Skipping Grad-CAM visualization due to load error.")
except Exception as e:
    print(f"âš ï¸ An unexpected error occurred during Grad-CAM visualization: {type(e).__name__}: {e}")
    print("   Skipping Grad-CAM visualization.")
finally:
    if cam_model is not None:
        del cam_model
    if device == "cuda":
        torch.cuda.empty_cache()

# Â§17 â€” Predict a NEW folder of images with the best model â†’ Excel
def predict_folder(folder, timm_name, threshold=0.5, out_xlsx="predictions.xlsx"):
    model = build_model(timm_name)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{timm_name}.pt"), map_location=device))
    model.eval()

    files, preds, confs = [], [], []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(VALID_EXT):
            continue
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
            p = torch.softmax(model(x).float(), 1)[0, POS_IDX].item()
        files.append(f)
        preds.append(CLASS_NAMES[1] if p >= threshold else CLASS_NAMES[0])
        confs.append(round(p, 4))

    out = pd.DataFrame({"Filename": files, "Prediction": preds, "Confidence(Mastitis)": confs})
    path = os.path.join(OUT_DIR, out_xlsx)
    out.to_excel(path, index=False)
    n_m = (out["Prediction"] == "Mastitis").sum()
    print(f"âœ… {len(out)} images â†’ {path}  |  Mastitis: {n_m}  Healthy: {len(out)-n_m}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out

# Example â€” point this at any folder of new cow images:
# df_new = predict_folder("/content/drive/MyDrive/mastitis_dataset/Serial_images",
#                         MODEL_ZOO[best_name])
# df_new.head()
print("Ready. Uncomment the last lines and set your folder path to run inference.")

for name in results:
    print(f"\n=== {name} ===")
    bootstrap_ci(results[name]["y_true"], results[name]["y_prob"])

# Â§18 â€” Build a pooled dataset (all images) tagged with cow-group IDs
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths); self.labels = list(labels); self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])

all_paths, all_labels, all_groups = [], [], []
for lbl, cls in enumerate(CLASS_NAMES):
    d = os.path.join(ROOT, cls)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(VALID_EXT):
            all_paths.append(os.path.join(d, f))
            all_labels.append(lbl)
            all_groups.append(f"{cls}__{cow_id_from_name(f)}")

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups)

print(f"Pooled images : {len(all_paths)}")
print(f"Unique cows   : {len(set(all_groups))}")
print(f"Class balance : {Counter(all_labels.tolist())}  (0=Healthy, 1=Mastitis)")

import gc, pickle
N_FOLDS   = 5
CV_EPOCHS = 12
CV_MODELS = [ "ViT-B/16", "Swin-Tiny"]

def cv_path(timm_name):
    return os.path.join(OUT_DIR, f"cv_result_{timm_name}.pkl")

def make_loader_from(paths, labels, tf, shuffle):
    return DataLoader(PathDataset(paths, labels, tf), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

sgkf  = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(all_paths, all_labels, groups=all_groups))
METRIC_KEYS = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1"]

cv_results = {}
for name, timm_name in MODEL_ZOO.items():          # reload cached CV results
    p = cv_path(timm_name)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                cv_results[name] = pickle.load(f)
            print(f"â†©ï¸Ž  loaded cached CV: {name}")
        except OSError as e:
            print(f"âš ï¸ Error loading cached CV result for {name} from {p}: {e}")
            print(f"   This model's cross-validation will be re-run if it's in CV_MODELS.")
        except Exception as e:
            print(f"âš ï¸ Unexpected error loading cached CV result for {name} from {p}: {e}")
            print(f"   This model's cross-validation will be re-run if it's in CV_MODELS.")

for name in CV_MODELS:
    if name in cv_results:
        continue                                   # resume: skip finished models
    timm_name = MODEL_ZOO[name]
    per_fold, oof_true, oof_prob = [], [], []
    print("\n" + "=" * 68)
    print(f"ðŸ” 5-FOLD CV â€” {name}  ({timm_name})")
    print("=" * 68)

    for k, (tr_idx, te_idx) in enumerate(folds, 1):
        assert not (set(all_groups[tr_idx]) & set(all_groups[te_idx])), "cow leak (test)!"
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        s_tr, s_va = next(gss.split(all_paths[tr_idx], all_labels[tr_idx],
                                    groups=all_groups[tr_idx]))
        tr2, va2 = tr_idx[s_tr], tr_idx[s_va]
        assert not (set(all_groups[tr2]) & set(all_groups[va2])), "cow leak (val)!"

        tl  = make_loader_from(all_paths[tr2],   all_labels[tr2],   train_tf, True)
        vl  = make_loader_from(all_paths[va2],   all_labels[va2],   eval_tf,  False)
        tel = make_loader_from(all_paths[te_idx], all_labels[te_idx], eval_tf, False)

        set_seed()
        model = build_model(timm_name)
        model, _, _ = train_model(model, f"{name}-f{k}",
                                  tloader=tl, vloader=vl, epochs=CV_EPOCHS)
        yt, yp = predict(model, tel)
        m = compute_metrics(yt, yp)
        per_fold.append(m); oof_true.append(yt); oof_prob.append(yp)
        print(f"  fold {k}: Acc {m['Accuracy']:.3f} | AUC {m['AUC']:.3f} | "
              f"Sens {m['Sensitivity']:.3f} | Spec {m['Specificity']:.3f} | F1 {m['F1']:.3f}")

        del model, tl, vl, tel, yt, yp     # free memory each fold
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    agg = {key: (float(np.mean([f[key] for f in per_fold])),
                 float(np.std ([f[key] for f in per_fold]))) for key in METRIC_KEYS}
    cv_results[name] = {"per_fold": per_fold, "agg": agg,
                        "oof_true": np.concatenate(oof_true),
                        "oof_prob": np.concatenate(oof_prob)}
    with open(cv_path(timm_name), "wb") as f:      # checkpoint this model's CV
        pickle.dump(cv_results[name], f)
    print(f"  â–¶ mean AUC {agg['AUC'][0]:.3f} Â± {agg['AUC'][1]:.3f} | "
          f"mean Acc {agg['Accuracy'][0]:.3f} Â± {agg['Accuracy'][1]:.3f}")

print("\nâœ… Cross-validation complete.")

# ===== Ø§Ø³ØªØ±Ø¬Ø§Ø¹ Ø§Ù„Ù€ CV Ø§Ù„Ù…Ø­ÙÙˆØ¸ Ù…Ù† Drive =====
import os, glob, pickle
cv_results = {}
name_by_timm = {v: k for k, v in MODEL_ZOO.items()}   # Ø§Ø³Ù… Ø§Ù„Ù…ÙˆØ¯ÙŠÙ„ Ù…Ù† Ø§Ø³Ù… timm
for f in sorted(glob.glob(os.path.join(OUT_DIR, "cv_result_*.pkl"))):
    timm_name = os.path.basename(f).replace("cv_result_", "").replace(".pkl", "")
    name = name_by_timm.get(timm_name, timm_name)
    with open(f, "rb") as fh:
        cv_results[name] = pickle.load(fh)
    print("â†©ï¸Ž loaded:", name)
METRIC_KEYS = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1"]
print(f"\nâœ… Ø¬Ø§Ù‡Ø²: {len(cv_results)} Ù…ÙˆØ¯ÙŠÙ„Ø§Øª")

# Â§20 â€” CV leaderboard (mean Â± std, sorted by AUC) + save CSV
rows = []
for name  in cv_results:          # ÙŠÙ„ÙÙ‘ Ø¹Ù„Ù‰ Ø§Ù„Ù…Ø­ÙÙˆØ¸ Ø¨Ø³ (Ù…Ø´ Ù…Ø­ØªØ§Ø¬ CV_MODELS):
    agg = cv_results[name]["agg"]
    row = {"Model": name}
    for key in METRIC_KEYS:
        mu, sd = agg[key]
        row[key] = f"{mu:.3f} Â± {sd:.3f}"
    row["_auc"] = agg["AUC"][0]
    rows.append(row)

cv_df = (pd.DataFrame(rows)
         .sort_values("_auc", ascending=False)
         .drop(columns="_auc")
         .reset_index(drop=True))
cv_df.index += 1
cv_df.to_csv(os.path.join(OUT_DIR, "cv_comparison.csv"))
print("Saved:", os.path.join(OUT_DIR, "cv_comparison.csv"))
cv_order = cv_df["Model"].tolist()
print(f"ðŸ† Best (CV mean AUC): {cv_order[0]}\n")
cv_df

# Â§21 â€” Per-fold AUC distribution (boxplot) â€” shows stability across folds
box_data = [[f["AUC"] for f in cv_results[n]["per_fold"]] for n in cv_order]
plt.figure(figsize=(11, 6))
bp = plt.boxplot(box_data, labels=cv_order, showmeans=True, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_alpha(0.5)
plt.xticks(rotation=20, ha="right")
plt.ylabel("AUC per fold"); plt.ylim(0, 1.02)
plt.title("5-Fold Cross-Validation â€” AUC distribution per model")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cv_auc_boxplot.png"), dpi=200, bbox_inches="tight")
plt.show()

# Â§22 â€” Out-of-fold (OOF) ROC: every image predicted exactly once across folds
plt.figure(figsize=(8, 7))
for name in cv_order:
    yt, yp = cv_results[name]["oof_true"], cv_results[name]["oof_prob"]
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    plt.plot(fpr, tpr, label=f"{name} (OOF AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Out-of-Fold ROC â€” 5-Fold Cross-Validation")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cv_oof_roc.png"), dpi=200, bbox_inches="tight")
plt.show()

# ===== CONTINUATION 1: identify + reload best model =====
import os, torch, numpy as np

if "cv_results" in globals() and len(cv_results) > 0:
    best_name = max(cv_results, key=lambda n: cv_results[n]["agg"]["AUC"][0])
    print(f"ðŸ† Best by 5-fold CV: {best_name} (AUC={cv_results[best_name]['agg']['AUC'][0]:.3f})")
else:
    best_name = max(results, key=lambda n: results[n]["metrics"]["AUC"])
    print(f"ðŸ† Best by single split: {best_name} (AUC={results[best_name]['metrics']['AUC']:.3f})")

best_timm = MODEL_ZOO[best_name]
ckpt = os.path.join(OUT_DIR, f"{best_timm}.pt")
if not os.path.exists(ckpt):                       # fall back to any trained model
    best_name = max(results, key=lambda n: results[n]["metrics"]["AUC"])
    best_timm = MODEL_ZOO[best_name]
    ckpt = os.path.join(OUT_DIR, f"{best_timm}.pt")

best_model = build_model(best_timm)
best_model.load_state_dict(torch.load(ckpt, map_location=device))
best_model.eval()
print("Loaded:", ckpt)

# ===== CONTINUATION 2: confusion matrix for the best model =====
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# use out-of-fold predictions if CV was run (most representative), else the test split
if "cv_results" in globals() and best_name in cv_results:
    yt, yp = cv_results[best_name]["oof_true"], cv_results[best_name]["oof_prob"]
    tag = "5-fold OOF"
else:
    yt, yp = results[best_name]["y_true"], results[best_name]["y_prob"]
    tag = "test split"

pred = (yp >= 0.5).astype(int)
cm  = confusion_matrix(yt, pred, labels=[0, 1])
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[0])
ax[0].set_title(f"{best_name} â€” Counts ({tag})"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[1])
ax[1].set_title("Normalized (per true class)"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"confusion_best_{best_timm}.png"), dpi=200, bbox_inches="tight")
plt.show()

print(classification_report(yt, pred, target_names=CLASS_NAMES, digits=3))

# ===== CONTINUATION 3: SHAP explainability (pixel attributions) =====
import shap, numpy as np, torch

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
def denorm(t):
    return np.clip(t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)

try:
    N_BG, N_EX, NSAMPLES = 10, 4, 30        # â†“ these if you hit OOM
    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(train_ds), size=min(N_BG, len(train_ds)), replace=False)
    background = torch.stack([train_ds[i][0] for i in bg_idx]).to(device)

    ex_imgs, seen = [], {0: 0, 1: 0}
    for x, y in test_ds:
        if seen[y] < N_EX // 2:
            ex_imgs.append(x); seen[y] += 1
        if len(ex_imgs) >= N_EX:
            break
    ex = torch.stack(ex_imgs).to(device)

    explainer = shap.GradientExplainer(best_model, background)
    sv = explainer.shap_values(ex, nsamples=NSAMPLES)

    # normalize shap output shape -> list of (N,H,W,C)
    if isinstance(sv, list):
        arrs = sv
    else:
        sv = np.array(sv)
        arrs = [sv[..., k] for k in range(sv.shape[-1])] if sv.ndim == 5 else [sv]
    shap_num = [np.transpose(a, (0, 2, 3, 1)) for a in arrs]
    disp = np.stack([denorm(t) for t in ex])

    shap.image_plot(shap_num, disp, show=True)
    print("Red = pushes toward the class; blue = pushes away.")
except Exception as e:
    print("âš ï¸ SHAP failed:", e)
    print("Try lowering N_BG/N_EX/NSAMPLES, or set best_model to a CNN "
          "(ResNet/DenseNet) â€” transformers are heavier for SHAP. Grad-CAM (Â§16) still works.")

# ===== CONTINUATION 4: inference on /content/drive/MyDrive/Serial_images =====
import os, pandas as pd, torch
from PIL import Image

SERIAL_DIR = os.path.join(ROOT, "Serial_images", "Serial_images")
try: VALID_EXT
except NameError: VALID_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

@torch.no_grad()
def dl_predict_image(path, model, threshold=0.5):
    x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        p = torch.softmax(model(x).float(), 1)[0, POS_IDX].item()
    return (CLASS_NAMES[1] if p >= threshold else CLASS_NAMES[0]), p

def dl_predict_folder(folder, model, threshold=0.5, out_xlsx="serial_preds.xlsx"):
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(VALID_EXT):
            continue
        if "..bmp" in f.lower():                    # skip malformed "73..bmp" names
            print(f"â­ï¸ Skipping {f}")
            continue
        pred, p = dl_predict_image(os.path.join(folder, f), model, threshold)
        rows.append({"Filename": f, "Prediction": pred, "Confidence(Mastitis)": round(p, 4)})
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, out_xlsx)
    df.to_excel(out, index=False)
    n_m = (df["Prediction"] == "Mastitis").sum()
    print(f"âœ… {len(df)} images â†’ {out} | Mastitis {n_m} / Healthy {len(df) - n_m}")
    return df

# single image (uncomment and set a real filename):
# print(dl_predict_image(os.path.join(SERIAL_DIR, "73.bmp"), best_model))

df_serial = dl_predict_folder(SERIAL_DIR, best_model)
df_serial.head(20)

# ===== SELF-CONTAINED: DL predictions + merge + honest SCC test =====
import os, re, glob, torch, numpy as np, pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# ---- SET THESE TWO PATHS ----
XLSX_PATH  = os.path.join(ROOT, "Mastitis_All_Comp", "blood parameters& ML prediction.xlsx")
IMG_FOLDERS = [os.path.join(ROOT, "Mastitis_All_Comp"),
               os.path.join(ROOT, "Serial_images", "Serial_images")]
SHEET = 0    # 0 = first sheet; change to "Sheet2" if SCC/CMT live there
# -----------------------------

try: VALID_EXT
except NameError: VALID_EXT = (".bmp",".png",".jpg",".jpeg",".tif",".tiff")

# 1) reload best model (DenseNet-121 = your best CV result)
best_name = "DenseNet-121"; best_timm = MODEL_ZOO[best_name]
best_model = build_model(best_timm)
best_model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{best_timm}.pt"), map_location=device))
best_model.eval()
print("âœ” model:", best_name)

# 2) predict image folders
@torch.no_grad()
def predict_img(path):
    x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    return torch.softmax(best_model(x).float(), 1)[0, POS_IDX].item()

rows = []
for folder in IMG_FOLDERS:
    if not os.path.isdir(folder):
        print("âš ï¸ missing:", folder); continue
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(VALID_EXT) or "..bmp" in f.lower():
            continue
        p = predict_img(os.path.join(folder, f))
        digits = re.findall(r"\d+", os.path.splitext(f)[0])
        rows.append({"stem": int(digits[0]) if digits else None,
                     "DL_pred": "Mastitis" if p >= 0.5 else "Healthy",
                     "DL_conf": round(p, 4)})
dl_preds = pd.DataFrame(rows).dropna(subset=["stem"]).drop_duplicates("stem")
print(f"âœ” predicted {len(dl_preds)} images")

# 3) read Excel + clean SCC
xls = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
xls.columns = [str(c).strip() for c in xls.columns]
print("Columns:", list(xls.columns))

def clean_scc(s):
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"<90000":"45000",">9000000":"9000000","":np.nan,"nan":np.nan}),
                         errors="coerce")

SCC_COL = next((c for c in xls.columns if "cell count" in c.lower() or c.lower()=="scc"), None)
CMT_COL = next((c for c in xls.columns if "california" in c.lower()), None)
ML_COL  = next((c for c in xls.columns if "prediction" in c.lower()), None)
if SCC_COL is None:
    raise ValueError("No SCC column found â€” check SHEET or column names above.")
xls[SCC_COL] = clean_scc(xls[SCC_COL])

# 4) auto-match ID column to filenames, then merge
best_id, best_hits = None, -1
for c in [c for c in xls.columns if "no" in c.lower()]:
    hits = pd.to_numeric(xls[c], errors="coerce").isin(dl_preds["stem"]).sum()
    if hits > best_hits: best_id, best_hits = c, hits
print(f"â†’ matching on '{best_id}' ({best_hits} matches)")
xls["_id"] = pd.to_numeric(xls[best_id], errors="coerce")
merged = xls.merge(dl_preds, left_on="_id", right_on="stem", how="inner")
print(f"âœ” merged rows: {len(merged)}")

enc = lambda s,k: s.astype(str).str.lower().str.contains(k).astype(int)
if CMT_COL: merged["CMT_bin"] = enc(merged[CMT_COL], "positive")
if ML_COL:  merged["ML_bin"]  = enc(merged[ML_COL],  "mastitis")
merged["DL_bin"] = (merged["DL_pred"]=="Mastitis").astype(int)

# 5) HONEST SCC significance (no forcing)
print("\n" + "="*55)
print("Does SCC separate healthy vs mastitis? (Mann-Whitney)")
print("="*55)
d = merged.dropna(subset=[SCC_COL]).copy()
scc = d[SCC_COL].astype(float).values
def scc_sig(label, col):
    if col not in d: print(f"[{label}] missing â€” skip"); return
    y = d[col].values; g0,g1 = scc[y==0], scc[y==1]
    if len(g0)<3 or len(g1)<3: print(f"[{label}] too few â€” skip"); return
    _, p = mannwhitneyu(g1, g0, alternative="greater")
    print(f"[{label}] median healthy={np.median(g0):,.0f} mastitis={np.median(g1):,.0f} "
          f"| p={p:.4g} â†’ {'SIGNIFICANT' if p<0.05 else 'ns'} | SCC AUC={roc_auc_score(y,scc):.3f}")
for lbl,col in [("vs CMT","CMT_bin"),("vs DL","DL_bin"),("vs old-ML","ML_bin")]:
    scc_sig(lbl, col)

os.makedirs(os.path.join(ROOT, "Mastitis_All_Comp"), exist_ok=True)
merged.to_excel(os.path.join(ROOT, "Mastitis_All_Comp", "merged_DL_SCC.xlsx"), index=False)
print("\nðŸ’¾ saved merged_DL_SCC.xlsx")
merged.head(10)

# ===== CONTINUATION 5: best DL model â†’ predict image folders =====
import os, re, torch, numpy as np, pandas as pd
from PIL import Image

try: VALID_EXT
except NameError: VALID_EXT = (".bmp",".png",".jpg",".jpeg",".tif",".tiff")

@torch.no_grad()
def dl_predict_image(path, model, threshold=0.5):
    x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        p = torch.softmax(model(x).float(), 1)[0, POS_IDX].item()
    return (CLASS_NAMES[1] if p >= threshold else CLASS_NAMES[0]), p

def dl_predict_folder(folder, model, threshold=0.5):
    if not os.path.isdir(folder):
        print("âš ï¸ not found:", folder); return pd.DataFrame(columns=["Filename","stem","DL_pred","DL_conf"])
    rows=[]
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(VALID_EXT) or "..bmp" in f.lower():
            continue
        pred,p = dl_predict_image(os.path.join(folder,f), model, threshold)
        digits = re.findall(r"\d+", os.path.splitext(f)[0])
        rows.append({"Filename":f, "stem":int(digits[0]) if digits else None,
                     "DL_pred":pred, "DL_conf":round(p,4)})
    df = pd.DataFrame(rows); print(f"  {folder}: {len(df)} images")
    return df

FOLDERS = [os.path.join(ROOT, "Mastitis_All_Comp"),
           os.path.join(ROOT, "Serial_images", "Serial_images")]
frames = [dl_predict_folder(fp, best_model) for fp in FOLDERS]
dl_preds = pd.concat([d for d in frames if len(d)], ignore_index=True) if any(len(d) for d in frames) else pd.DataFrame()
dl_preds = dl_preds.drop_duplicates(subset="stem", keep="first")

os.makedirs(os.path.join(ROOT, "Mastitis_All_Comp"), exist_ok=True)
dl_preds.to_excel(os.path.join(ROOT, "Mastitis_All_Comp", "DL_predictions.xlsx"), index=False)
print(f"\nTotal: {len(dl_preds)} | Mastitis={(dl_preds['DL_pred']=='Mastitis').sum()}  Healthy={(dl_preds['DL_pred']=='Healthy').sum()}")
dl_preds.head()

# ===== CONTINUATION 6: merge DL with the Excel, add DL-best column =====
import pandas as pd, numpy as np

XLSX_PATH = os.path.join(ROOT, "Mastitis_All_Comp", "blood parameters& ML prediction.xlsx")
SHEET     = "Sheet1"   # per your screenshot, Sheet1 holds Animal no./Prediction/CMT/SCC

xls = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
xls.columns = [str(c).strip() for c in xls.columns]
print("Columns:", list(xls.columns))

def clean_scc(s):
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"<90000":"45000", ">9000000":"9000000", "":np.nan, "nan":np.nan}),
                         errors="coerce")

SCC_COL = next((c for c in xls.columns if "cell count" in c.lower() or c.lower()=="scc"), None)
CMT_COL = next((c for c in xls.columns if "california" in c.lower()), None)
ML_COL  = next((c for c in xls.columns if "prediction" in c.lower()), None)
if SCC_COL: xls[SCC_COL] = clean_scc(xls[SCC_COL])

# auto-pick the ID column that matches the most image filenames
best_id, best_hits = None, -1
for c in [c for c in xls.columns if "no" in c.lower()]:
    hits = pd.to_numeric(xls[c], errors="coerce").isin(dl_preds["stem"]).sum()
    print(f"  match on '{c}': {hits}")
    if hits > best_hits: best_id, best_hits = c, hits
print("â†’ matching on:", best_id)

xls["_id"] = pd.to_numeric(xls[best_id], errors="coerce")
merged = xls.merge(dl_preds[["stem","DL_pred","DL_conf"]], left_on="_id", right_on="stem", how="inner")
print(f"Matched rows: {len(merged)}")

enc = lambda s,k: s.astype(str).str.lower().str.contains(k).astype(int)
if CMT_COL: merged["CMT_bin"] = enc(merged[CMT_COL], "positive")
if ML_COL:  merged["ML_bin"]  = enc(merged[ML_COL],  "mastitis")
merged["DL_bin"] = (merged["DL_pred"]=="Mastitis").astype(int)
merged[f"DL_best ({best_name})"] = merged["DL_pred"]     # â† the requested column

keep = [best_id, SCC_COL, CMT_COL, ML_COL, f"DL_best ({best_name})", "DL_conf"]
comp = merged[[c for c in keep if c in merged.columns]].copy()
comp_path = os.path.join(ROOT, "Mastitis_All_Comp", "Comparison_DL_vs_CMT_SCC_ML.xlsx")
comp.to_excel(comp_path, index=False); print("Saved:", comp_path)
comp.head(20)

# ===== CONTINUATION 7: agreement metrics + confusion matrices =====
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

def agree(name, ref, hat):
    tn,fp,fn,tp = confusion_matrix(ref, hat, labels=[0,1]).ravel()
    return {"comparison":name, "n":len(ref),
            "Accuracy":accuracy_score(ref,hat),
            "Sensitivity":tp/(tp+fn) if tp+fn else np.nan,
            "Specificity":tn/(tn+fp) if tn+fp else np.nan,
            "Kappa":cohen_kappa_score(ref,hat)}

rows=[]
if "CMT_bin" in merged:
    rows.append(agree("DL vs CMT",      merged["CMT_bin"], merged["DL_bin"]))
    if "ML_bin" in merged: rows.append(agree("old-ML vs CMT", merged["CMT_bin"], merged["ML_bin"]))
if "ML_bin" in merged:  rows.append(agree("DL vs old-ML",  merged["ML_bin"],  merged["DL_bin"]))
summary = pd.DataFrame(rows).round(3)
print(summary.to_string(index=False))
summary.to_excel(os.path.join(ROOT, "Mastitis_All_Comp", "agreement_summary.xlsx"), index=False)

if "CMT_bin" in merged:
    pairs = [("DL vs CMT","DL_bin"), ("old-ML vs CMT","ML_bin")]
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    for a,(title,col) in zip(ax, pairs):
        if col not in merged: a.axis("off"); continue
        cm = confusion_matrix(merged["CMT_bin"], merged[col], labels=[0,1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=a)
        a.set_title(title); a.set_xlabel("Predicted"); a.set_ylabel("CMT (reference)")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "Mastitis_All_Comp", "confusion_vs_CMT.png"), dpi=200, bbox_inches="tight")
    plt.show()

# ===== CONTINUATION 8: HONEST SCC analysis (no p-value forcing) =====
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

d = merged.dropna(subset=[SCC_COL]).copy()
scc = d[SCC_COL].astype(float).values

def scc_report(label, ybin):
    y = d[ybin].values
    g0, g1 = scc[y==0], scc[y==1]
    if len(g0)<3 or len(g1)<3:
        print(f"[{label}] too few in one group (n0={len(g0)}, n1={len(g1)}) â€” skipped"); return
    _, p = mannwhitneyu(g1, g0, alternative="greater")     # SCC higher in mastitis?
    print(f"[{label}] median SCC  healthy={np.median(g0):,.0f}  mastitis={np.median(g1):,.0f} "
          f"| Mann-Whitney p={p:.4g} | SCC AUC={roc_auc_score(y, scc):.3f}")

print("SCC as a discriminator (non-parametric, honest):")
for lbl,col in [("vs CMT","CMT_bin"),("vs DL-best","DL_bin"),("vs old-ML","ML_bin")]:
    if col in d: scc_report(lbl, col)

def honest_logit(label, ybin):
    y = d[ybin].values.astype(float)
    X = sm.add_constant(np.log10(np.clip(scc, 1, None)))    # log10(SCC) is standard
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=300)
        sep = "  âš ï¸ likely separation" if abs(res.params[1])>50 else ""
        print(f"[{label}] log10-SCC coef={res.params[1]:.3f}  REAL p={res.pvalues[1]:.4g}{sep}")
    except Exception as e:
        print(f"[{label}] did not converge (perfect separation) â†’ use the Mann-Whitney result. {type(e).__name__}")

print("\nLogistic regression on log10(SCC) â€” real p-values:")
for lbl,col in [("CMT","CMT_bin"),("DL-best","DL_bin"),("old-ML","ML_bin")]:
    if col in d: honest_logit(lbl, col)

# ===== Full explanation: image counts + Confusion Matrix breakdown =====
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.metrics import confusion_matrix

# â”€â”€ 1) Auto-pick the best model â”€â”€
if "cv_results" in globals() and len(cv_results) > 0:
    best_name = max(cv_results, key=lambda n: cv_results[n]["agg"]["AUC"][0])
    y_true = cv_results[best_name]["oof_true"]
    y_prob = cv_results[best_name]["oof_prob"]
    source = "Cross-Validation (each image predicted once by a fold that never saw its cow)"
else:
    best_name = max(results, key=lambda n: results[n]["metrics"]["AUC"])
    y_true = results[best_name]["y_true"]
    y_prob = results[best_name]["y_prob"]
    source = "Single split (held-out test set)"

y_pred = (y_prob >= 0.5).astype(int)

# â”€â”€ 2) Image counts at each stage â”€â”€
print("=" * 62)
print(f"ðŸ† Best model : {best_name}")
print(f"ðŸ“Œ Results from: {source}")
print("=" * 62)

try:
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    print("\nðŸ“‚ Images in the base split (Part 1):")
    print(f"   â€¢ Train : {n_train} images")
    print(f"   â€¢ Val   : {n_val} images")
    print(f"   â€¢ Test  : {n_test} images")
    print(f"   â€¢ Total : {n_train + n_val + n_test} images")
except Exception:
    pass

n_eval = len(y_true)
n_healthy = int((y_true == 0).sum())
n_mastitis = int((y_true == 1).sum())
print(f"\nðŸ”¬ Images the model was evaluated on for THIS result: {n_eval}")
print(f"   â€¢ Healthy  : {n_healthy}")
print(f"   â€¢ Mastitis : {n_mastitis}")

# â”€â”€ 3) Confusion matrix â”€â”€
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 62)
print("ðŸ“Š Confusion Matrix")
print("=" * 62)
print(f"""
                          â”‚ Predicted Healthy â”‚ Predicted Mastitis â”‚
 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
  Actual: Healthy         â”‚   {tn:^15}   â”‚   {fp:^16}   â”‚
  Actual: Mastitis        â”‚   {fn:^15}   â”‚   {tp:^16}   â”‚
""")

print("ðŸ”Ž What each cell means:")
print(f"   âœ… TN = {tn}: cow is Healthy and model said Healthy   â†’ correct")
print(f"   âœ… TP = {tp}: cow is Mastitis and model said Mastitis â†’ correct")
print(f"   âŒ FP = {fp}: cow is Healthy but model said Mastitis  â†’ false alarm")
print(f"   âŒ FN = {fn}: cow is Mastitis but model said Healthy  â†’ dangerous (missed a sick cow)")

# â”€â”€ 4) Metrics with interpretation â”€â”€
acc  = (tp + tn) / n_eval
sens = tp / (tp + fn) if (tp + fn) else 0
spec = tn / (tn + fp) if (tn + fp) else 0
prec = tp / (tp + fp) if (tp + fp) else 0
npv  = tn / (tn + fn) if (tn + fn) else 0
f1   = 2 * prec * sens / (prec + sens) if (prec + sens) else 0

print("\n" + "=" * 62)
print("ðŸ“ˆ Metrics & interpretation")
print("=" * 62)
print(f"â€¢ Accuracy = {acc:.3f}")
print(f"    Share of all predictions that were correct ({tp+tn} of {n_eval}).")
print(f"\nâ€¢ Sensitivity (Recall) = {sens:.3f}")
print(f"    Of all truly Mastitis cows, the model caught {sens*100:.1f}%.")
print(f"    â† Most important for disease screening: you want this high so you don't miss cases.")
print(f"\nâ€¢ Specificity = {spec:.3f}")
print(f"    Of all truly Healthy cows, the model correctly cleared {spec*100:.1f}%.")
print(f"\nâ€¢ Precision (PPV) = {prec:.3f}")
print(f"    When the model says 'Mastitis', it is right {prec*100:.1f}% of the time.")
print(f"\â€¢ NPV = {npv:.3f}")
print(f"    When the model says 'Healthy', it is right {npv*100:.1f}% of the time.")
print(f"\nâ€¢ F1-Score = {f1:.3f}")
print(f"    Balance between Precision and Sensitivity (one number summarizing both).")

# â”€â”€ 5) Plot the confusion matrix (counts + normalized) â”€â”€
labels = ["Healthy", "Mastitis"]
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax[0], annot_kws={"size": 14})
ax[0].set_title(f"{best_name} â€” Counts", fontsize=12)
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")

sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens", cbar=False, vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax[1], annot_kws={"size": 14})
ax[1].set_title("Normalized (per actual row)", fontsize=12)
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"confusion_explained_{best_name}.png"), dpi=200, bbox_inches="tight")
plt.show()

print(f"\nðŸ’¾ Figure saved to: {OUT_DIR}")

# ===== Grad-CAM Explainability (Ø§Ù„Ø¨ÙŠØ¨Ø± Ù…Ø¹Ù…Ù„ØªÙ‡Ø§Ø´ â€” Ø¯ÙŠ Ù†Ù‚Ø·Ø© ØªÙ…ÙŠÙ‘Ø²) =====
import os, torch, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# Ù„Ùˆ Ù…Ø´ Ù…ØªØ³Ø·Ø¨Ø©: !pip -q install grad-cam

CAM_MODEL_NAME = "DenseNet-121"          # Ø£ÙØ¶Ù„ CNN Ø¹Ù†Ø¯Ùƒ (Ø¨ÙŠØ·Ù„Ø¹ heatmap Ù†Ø¶ÙŠÙ)
cam_timm = MODEL_ZOO[CAM_MODEL_NAME]
cam_model = build_model(cam_timm)
cam_model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{cam_timm}.pt"), map_location=device))
cam_model.eval()

def last_conv_layer(model):
    conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            conv = m
    return conv                          # Ø¢Ø®Ø± Conv2d = Ø§Ù„Ø·Ø¨Ù‚Ø© Ø§Ù„ØµØ­ Ù„Ù„Ù€ CAM ÙÙŠ Ø§Ù„Ù€ CNNs

def denorm(t):
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(x, 0, 1)

# Ø§Ø®ØªØ§Ø± 6 ØµÙˆØ± Ù…Ù† Ø§Ù„ØªØ³Øª (3 Ø³Ù„ÙŠÙ…Ø© + 3 Ù…Ø±ÙŠØ¶Ø©)
samples, seen = [], {0: 0, 1: 0}
for path, lbl in test_ds.samples:
    if seen[lbl] < 3:
        samples.append((path, lbl)); seen[lbl] += 1
    if len(samples) >= 6:
        break

cam = GradCAM(model=cam_model, target_layers=[last_conv_layer(cam_model)])
plt.figure(figsize=(15, 6))
for i, (path, lbl) in enumerate(samples[:6]):
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(cam_model(x).float(), 1)[0, POS_IDX].item()
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(POS_IDX)])[0]
    vis = show_cam_on_image(denorm(x[0]), grayscale, use_rgb=True)
    plt.subplot(2, 3, i + 1)
    plt.imshow(vis); plt.axis("off")
    plt.title(f"True: {CLASS_NAMES[lbl]} | P(mastitis)={prob:.2f}", fontsize=10)
plt.suptitle(f"Grad-CAM Explainability â€” {CAM_MODEL_NAME}", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"gradcam_{cam_timm}.png"), dpi=200, bbox_inches="tight")
plt.show()

del cam_model
if device == "cuda":
    torch.cuda.empty_cache()
print("ðŸ’¾ Ø§ØªØ­ÙØ¸Øª ÙÙŠ:", OUT_DIR)

# ===== SCC significance â€” HONEST (Ø²ÙŠ Ø£Ù…Ø§Ù†Ø© Ø§Ù„Ø¨ÙŠØ¨Ø±ØŒ Ø¨Ø¯ÙˆÙ† forcing) =====
import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

XLSX_PATH = os.path.join(ROOT, "Mastitis_All_Comp", "blood parameters& ML prediction.xlsx")
SHEET = 0    # ØºÙŠÙ‘Ø±Ù‡Ø§ Ù„Ù€ "Sheet2" Ù„Ùˆ Ø§Ù„Ù€ SCC/CMT ÙÙŠ Ø§Ù„Ø´ÙŠØª Ø§Ù„ØªØ§Ù†ÙŠ

xls = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
xls.columns = [str(c).strip() for c in xls.columns]
print("Columns found:", list(xls.columns))

def clean_scc(s):                        # Ù†ÙØ³ ØªÙ†Ø¸ÙŠÙ Ø§Ù„Ø¨ÙŠØ¨Ø± Ø¨Ø§Ù„Ø¸Ø¨Ø·
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"<90000": "45000", ">9000000": "9000000",
                                    "": np.nan, "nan": np.nan}), errors="coerce")

SCC_COL = next((c for c in xls.columns if "cell count" in c.lower() or c.lower() == "scc"), None)
CMT_COL = next((c for c in xls.columns if "california" in c.lower()), None)
ML_COL  = next((c for c in xls.columns if "prediction" in c.lower()), None)
if SCC_COL is None:
    raise ValueError("Ù…ÙÙŠØ´ Ø¹Ù…ÙˆØ¯ SCC â€” ØºÙŠÙ‘Ø± SHEET Ø£Ùˆ Ø¨ÙØµ Ø¹Ù„Ù‰ Ø£Ø³Ù…Ø§Ø¡ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© ÙÙˆÙ‚.")
xls[SCC_COL] = clean_scc(xls[SCC_COL])

enc = lambda s, k: s.astype(str).str.lower().str.contains(k).astype(int)
if CMT_COL: xls["CMT_bin"] = enc(xls[CMT_COL], "positive")
if ML_COL:  xls["ML_bin"]  = enc(xls[ML_COL], "mastitis")

d = xls.dropna(subset=[SCC_COL]).copy()
scc_vals = d[SCC_COL].astype(float).values

def scc_report(label, ybin_col):
    if ybin_col not in d.columns:
        print(f"[{label}] Ø§Ù„Ø¹Ù…ÙˆØ¯ Ù…Ø´ Ù…ÙˆØ¬ÙˆØ¯ â€” ØªØ®Ø·Ù‘ÙŠ"); return
    y = d[ybin_col].values
    g0, g1 = scc_vals[y == 0], scc_vals[y == 1]
    if len(g0) < 3 or len(g1) < 3:
        print(f"[{label}] Ø¹ÙŠÙ†Ø§Øª Ù‚Ù„ÙŠÙ„Ø© (n0={len(g0)}, n1={len(g1)}) â€” ØªØ®Ø·Ù‘ÙŠ"); return
    stat, p = mannwhitneyu(g1, g0, alternative="greater")   # SCC Ø£Ø¹Ù„Ù‰ ÙÙŠ Ø§Ù„Ù…Ø±ÙŠØ¶Ø©ØŸ
    auc = roc_auc_score(y, scc_vals)
    verdict = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
    print(f"\n[{label}]  n0={len(g0)} n1={len(g1)}")
    print(f"   median SCC  healthy={np.median(g0):,.0f}  mastitis={np.median(g1):,.0f}")
    print(f"   Mann-Whitney U  p = {p:.5f}  -> {verdict}")
    print(f"   SCC discrimination AUC = {auc:.3f}")

print("\n" + "=" * 60)
print("SCC ÙƒÙ…Ù…ÙŠÙ‘Ø² Ø¨ÙŠÙ† Ø³Ù„ÙŠÙ…Ø©/Ù…Ø±ÙŠØ¶Ø© (Ø§Ø®ØªØ¨Ø§Ø± Ù„Ø§-Ù…Ø¹Ù„Ù…ÙŠØŒ Ø£Ù…ÙŠÙ†)")
print("=" * 60)
for lbl, col in [("SCC vs CMT", "CMT_bin"), ("SCC vs ML-prediction", "ML_bin")]:
    scc_report(lbl, col)

def honest_logit(label, ybin_col):
    if ybin_col not in d.columns:
        return
    y = d[ybin_col].values.astype(float)
    X = sm.add_constant(np.log10(np.clip(scc_vals, 1, None)))   # log10(SCC) Ù‡Ùˆ Ø§Ù„Ù…Ø¹ØªØ§Ø¯
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=300)
        flag = "  (Ø¹Ù„Ù‰ Ø§Ù„Ø£Ø±Ø¬Ø­ separation)" if abs(res.params[1]) > 50 else ""
        print(f"[{label}] log10-SCC coef={res.params[1]:.3f}  REAL p={res.pvalues[1]:.5f}{flag}")
    except Exception as e:
        print(f"[{label}] Ù…ÙÙŠØ´ convergence (perfect separation) â†’ Ø§Ø³ØªØ®Ø¯Ù… Mann-Whitney. {type(e).__name__}")

print("\n" + "=" * 60)
print("Logistic regression Ø¹Ù„Ù‰ log10(SCC) â€” p-values Ø­Ù‚ÙŠÙ‚ÙŠØ© (Ø¨Ø¯ÙˆÙ† ØªØ²ÙˆÙŠØ±)")
print("=" * 60)
for lbl, col in [("CMT", "CMT_bin"), ("ML-prediction", "ML_bin")]:
    honest_logit(lbl, col)

import numpy as np, pandas as pd, itertools, os
from scipy import stats

# ============================================================
#  Paired statistical comparison of models (reviewer point #2)
#  (A) DeLong test on pooled OOF predictions
#  (B) Nadeau-Bengio corrected resampled t-test on fold AUCs
#  Runs on cv_results. Add ViT/Swin once their CV is done.
# ============================================================
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m; k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r]); tz[r] = _compute_midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n

def delong_test(y_true, prob_a, prob_b):
    order = np.argsort(-y_true); y = y_true[order]; m = int(y.sum())
    preds = np.vstack([prob_a[order], prob_b[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0: return aucs[0], aucs[1], float("nan"), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * (1 - stats.norm.cdf(abs(z)))

def corrected_t_test(diffs, n_train, n_test):
    diffs = np.asarray(diffs, dtype=float); k = len(diffs)
    mean = diffs.mean(); var = diffs.var(ddof=1)
    if var == 0: return mean, float("nan"), float("nan")
    corr = (1.0 / k) + (n_test / n_train)
    t = mean / np.sqrt(corr * var)
    return mean, t, 2 * (1 - stats.t.cdf(abs(t), df=k - 1))

models = list(cv_results.keys())
print("Models compared:", models, "\n")

print("="*70); print("(A) DeLong test â€” AUC differences on out-of-fold predictions"); print("="*70)
rows = []
for a, b in itertools.combinations(models, 2):
    yt = cv_results[a]["oof_true"].astype(int)
    auc_a, auc_b, z, p = delong_test(yt, cv_results[a]["oof_prob"], cv_results[b]["oof_prob"])
    sig = "significant" if p < 0.05 else "n.s."
    rows.append([f"{a} vs {b}", round(auc_a,3), round(auc_b,3), round(p,4), sig])
    print(f"{a:>15} ({auc_a:.3f}) vs {b:<15} ({auc_b:.3f})  |  p = {p:.4f}  {sig}")
delong_df = pd.DataFrame(rows, columns=["Comparison","AUC_A","AUC_B","p_value","Result"])

print("\n" + "="*70); print("(B) Nadeau-Bengio corrected resampled t-test â€” fold-wise AUC"); print("="*70)
fold_auc = {m: [f["AUC"] for f in cv_results[m]["per_fold"]] for m in models}
N_TOTAL = len(cv_results[models[0]]["oof_true"]); n_test = N_TOTAL/5.0; n_train = N_TOTAL - n_test
rows2 = []
for a, b in itertools.combinations(models, 2):
    diffs = np.array(fold_auc[a]) - np.array(fold_auc[b])
    mean, t, p = corrected_t_test(diffs, n_train, n_test)
    sig = "significant" if (p == p and p < 0.05) else "n.s."
    rows2.append([f"{a} vs {b}", round(mean,4), round(p,4) if p==p else None, sig])
    print(f"{a:>15} vs {b:<15}  |  mean Î”AUC = {mean:+.4f}  p = {p:.4f}  {sig}")
ttest_df = pd.DataFrame(rows2, columns=["Comparison","mean_dAUC","p_value","Result"])

delong_df.to_csv(os.path.join(OUT_DIR, "significance_delong.csv"), index=False)
ttest_df.to_csv(os.path.join(OUT_DIR, "significance_corrected_ttest.csv"), index=False)
print("\nSaved to", OUT_DIR)
print("\nGuide: DeLong p<0.05 => AUCs differ significantly on the same images.")
print("If DenseNet-121 vs Inception-V3 is n.s., report them as statistically comparable.")

# ==========================================================================
# CELL A (STANDALONE) â€” Holmâ€“Bonferroni, self-contained
#   Reloads cv_results from Drive and defines DeLong inline, so it runs even
#   on a fresh runtime without Â§21/Â§22/Â§44.
# ==========================================================================
import os, glob, pickle, itertools, numpy as np, pandas as pd
from scipy import stats

OUT_DIR = os.path.join(ROOT, "DL_results")   # â† same as your Â§3 config

# 0) use the local output directory
os.makedirs(OUT_DIR, exist_ok=True)

# 1) load cv_results if not already in memory
try:
    cv_results
    print(f"cv_results already in memory: {list(cv_results.keys())}")
except NameError:
    MODEL_ZOO = {"ResNet-50":"resnet50","DenseNet-121":"densenet121",
                 "EfficientNet-B0":"efficientnet_b0","MobileNetV2":"mobilenetv2_100",
                 "Inception-V3":"inception_v3","ViT-B/16":"vit_base_patch16_224",
                 "Swin-Tiny":"swin_tiny_patch4_window7_224"}
    name_by_timm = {v: k for k, v in MODEL_ZOO.items()}
    cv_results = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "cv_result_*.pkl"))):
        timm_name = os.path.basename(f).replace("cv_result_", "").replace(".pkl", "")
        name = name_by_timm.get(timm_name, timm_name)
        with open(f, "rb") as fh:
            cv_results[name] = pickle.load(fh)
        print("â†©ï¸Ž loaded:", name)

assert len(cv_results) >= 2, (
    f"Only {len(cv_results)} CV result(s) found in {OUT_DIR}. "
    "Run the CV cells (Â§21) or the reload cell (Â§22) first, or fix OUT_DIR.")

# 2) inline DeLong (fast) + Holm
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1]-m; k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :m]); ty[r] = _midrank(preds_sorted[r, m:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(axis=1)/m - (m+1.0)/2.0)/n
    v01 = (tz[:, :m]-tx)/n; v10 = 1.0-(tz[:, m:]-ty)/m
    return aucs, np.cov(v01)/m + np.cov(v10)/n

def delong_test(y_true, pa, pb):
    order = np.argsort(-y_true); y = y_true[order]; m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack([pa[order], pb[order]]), m)
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    if var <= 0: return aucs[0], aucs[1], 1.0
    z = (aucs[0]-aucs[1])/np.sqrt(var)
    return aucs[0], aucs[1], 2*(1-stats.norm.cdf(abs(z)))

def holm_bonferroni(pvals, alpha=0.05):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m-rank)*p[idx]); adj[idx] = min(run, 1.0)
    return adj, adj < alpha

# 3) pre-specified family: best model vs the other six
best = max(cv_results, key=lambda n: cv_results[n]["agg"]["AUC"][0])
others = [m for m in cv_results if m != best]
print(f"\nReference (best) model: {best}\nCompared against: {others}\n")
yt = cv_results[best]["oof_true"].astype(int)

rows = []
for m in others:
    a_b, a_m, p = delong_test(yt, cv_results[best]["oof_prob"], cv_results[m]["oof_prob"])
    rows.append({"comparison": f"{best} vs {m}", "dAUC": round(a_b-a_m, 4), "DeLong_p_raw": p})
tbl = pd.DataFrame(rows)
tbl["DeLong_p_Holm"], tbl["sig_Holm"] = holm_bonferroni(tbl["DeLong_p_raw"].values)

# 4) corrected-t on fold AUCs, also Holm-adjusted
fold_auc = {m: [f["AUC"] for f in cv_results[m]["per_fold"]] for m in cv_results}
N = len(yt); n_te = N/5.0; n_tr = N-n_te
ct = []
for m in others:
    d = np.array(fold_auc[best])-np.array(fold_auc[m]); k = len(d); v = d.var(ddof=1)
    p = 2*(1-stats.t.cdf(abs(d.mean()/np.sqrt((1.0/k+n_te/n_tr)*v)), df=k-1)) if v > 0 else np.nan
    ct.append(p)
tbl["ct_p_raw"] = ct
tbl["ct_p_Holm"], _ = holm_bonferroni(np.nan_to_num(ct, nan=1.0))

pd.set_option("display.width", 160)
print(tbl.round(4).to_string(index=False))
tbl.round(6).to_csv(os.path.join(OUT_DIR, "significance_holm.csv"), index=False)
print(f"\nðŸ’¾ saved: {os.path.join(OUT_DIR, 'significance_holm.csv')}")


# ==========================================================================
# CELL B â€” Fixed per-model COLOURS + Precisionâ€“Recall curves (Supp. Fig. S3)
#   * MODEL_COLORS gives every architecture ONE colour used in every figure
#     (reviewer figure note). Re-run your Â§11/Â§12/Â§21/Â§22 plots passing
#     color=MODEL_COLORS[name] so DenseNet-121 is the same colour throughout.
#   * PR curves are more informative than ROC under class imbalance.
# ==========================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# one stable colour per architecture (order = your MODEL_ZOO)
_ARCH_ORDER = list(MODEL_ZOO.keys())
_palette = plt.cm.tab10(np.linspace(0, 1, 10))
MODEL_COLORS = {name: _palette[i % 10] for i, name in enumerate(_ARCH_ORDER)}
print("Fixed colours (reuse everywhere):")
for k, v in MODEL_COLORS.items():
    print(f"  {k:<16} -> RGBA {tuple(round(float(c),3) for c in v)}")

# PR curves â€” prefer OOF (each image scored once by a blind fold); else held-out
def _pr_source(name):
    if "cv_results" in globals() and name in cv_results:
        return cv_results[name]["oof_true"], cv_results[name]["oof_prob"], "OOF (5-fold)"
    if "results" in globals() and name in results:
        return results[name]["y_true"], results[name]["y_prob"], "held-out"
    return None, None, None

plt.figure(figsize=(7.5, 6.5))
src_label = None
for name in _ARCH_ORDER:
    yt_, yp_, src = _pr_source(name)
    if yt_ is None:
        continue
    src_label = src
    prec, rec, _ = precision_recall_curve(yt_, yp_, pos_label=1)
    ap = average_precision_score(yt_, yp_)
    plt.plot(rec, prec, color=MODEL_COLORS[name], lw=2, label=f"{name} (AP={ap:.3f})")
# baseline = prevalence of the positive (mastitis) class
_prev = float(np.mean(np.concatenate([_pr_source(n)[0] for n in _ARCH_ORDER
                                       if _pr_source(n)[0] is not None])))
plt.axhline(_prev, ls="--", color="grey", lw=1, label=f"chance (prevalence={_prev:.2f})")
plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision (PPV)")
plt.title(f"Precisionâ€“Recall curves â€” {src_label}")
plt.legend(loc="upper right", fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout()
_pr_png = os.path.join(OUT_DIR, "precision_recall_curves.png")
plt.savefig(_pr_png, dpi=200, bbox_inches="tight"); plt.show()
print(f"\nðŸ’¾ saved Supplementary Figure S3: {_pr_png}")


# ==========================================================================
# CELL C â€” Per-model bootstrap 95% CIs on the HELD-OUT set (Supp. Table S1)
#   Reviewer point: Table 3 has no uncertainty. This exports CIs for every
#   model & metric so you can add them to the paper / supplement.
# ==========================================================================
import numpy as np, pandas as pd, os

def _boot_ci(y_true, y_prob, n_boot=1000, thr=0.5, seed=42):
    rng = np.random.default_rng(seed)
    keys = ["Accuracy", "AUC", "Sensitivity", "Specificity", "Precision", "F1"]
    store = {k: [] for k in keys}
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], y_prob[idx], thr)
        for k in keys:
            store[k].append(m.get(k, np.nan))
    out = {}
    for k in keys:
        a = np.array(store[k], float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        out[k] = (round(float(np.nanmean(a)), 4), round(float(lo), 4), round(float(hi), 4))
    return out

rows = []
_src = results if ("results" in globals() and len(results)) else None
if _src is None:
    print("âš ï¸ `results` (held-out predictions) not in memory â€” run Part 1 first.")
else:
    for name in _src:
        yt_ = _src[name].get("y_true"); yp_ = _src[name].get("y_prob")
        if yt_ is None:
            continue
        ci = _boot_ci(yt_, yp_)
        row = {"Model": name}
        for k, (mean, lo, hi) in ci.items():
            row[k] = f"{mean:.3f} ({lo:.3f}â€“{hi:.3f})"
        rows.append(row)
    s1 = pd.DataFrame(rows)
    print(s1.to_string(index=False))
    _s1_path = os.path.join(OUT_DIR, "held_out_bootstrap_CIs_TableS1.csv")
    s1.to_csv(_s1_path, index=False)
    print(f"\nðŸ’¾ saved Supplementary Table S1: {_s1_path}")


# ==========================================================================
# CELL D â€” LEAKAGE-FREE OOF predictions for the SCC / CMT analyses
#          + explicit 77-vs-85 flow count   (reviewer points #4 and 77â†”85)
#   Builds DenseNet-121 out-of-fold predictions (each cow scored by a fold
#   that never saw it), aggregates to one prediction per cow, merges with the
#   blood/SCC sheet, and prints the exact counts at every filtering step.
#   USE THIS `merged_oof` for your Â§38/Â§40 agreement + SCC cells instead of
#   the single-split model, so Tables 8â€“9 are leakage-free.
# ==========================================================================
import os, re, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, accuracy_score

XLSX_PATH = os.path.join(ROOT, "Mastitis_All_Comp", "blood parameters& ML prediction.xlsx")
SHEET     = 0

best = max(cv_results, key=lambda n: cv_results[n]["agg"]["AUC"][0])
print(f"Using out-of-fold predictions of: {best}\n")

# 1) reconstruct OOF path order: folds are deterministic, OOF was concatenated
#    in fold order over each fold's TEST indices (shuffle=False loaders).
oof_order = np.concatenate([te_idx for (_tr, te_idx) in folds])
oof_paths = np.asarray(all_paths)[oof_order]
oof_prob  = np.asarray(cv_results[best]["oof_prob"], float)
oof_true  = np.asarray(cv_results[best]["oof_true"], int)
assert len(oof_paths) == len(oof_prob), "OOF length mismatch â€” re-run the CV cell first."

def _stem(path):
    d = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(d[0]) if d else None

img_df = pd.DataFrame({"stem": [_stem(p) for p in oof_paths],
                       "oof_prob": oof_prob, "true": oof_true}).dropna(subset=["stem"])
img_df["stem"] = img_df["stem"].astype(int)

# 2) aggregate per COW (mean OOF prob over that cow's images) -> per-cow label
cow = (img_df.groupby("stem")
              .agg(DL_conf=("oof_prob", "mean"), n_img=("oof_prob", "size"))
              .reset_index())
cow["DL_pred"] = np.where(cow["DL_conf"] >= 0.5, "Mastitis", "Healthy")
cow["DL_bin"]  = (cow["DL_pred"] == "Mastitis").astype(int)
print(f"Cows with OOF predictions: {len(cow)} (from {len(img_df)} images)")

# 3) merge with the blood/SCC sheet
xls = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
xls.columns = [str(c).strip() for c in xls.columns]

def clean_scc(s):
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"<90000": "45000", ">9000000": "9000000",
                                    "": np.nan, "nan": np.nan}), errors="coerce")

SCC_COL = next((c for c in xls.columns if "cell count" in c.lower() or c.lower() == "scc"), None)
CMT_COL = next((c for c in xls.columns if "california" in c.lower()), None)
ML_COL  = next((c for c in xls.columns if "prediction" in c.lower()), None)
if SCC_COL: xls["_SCC_num"] = clean_scc(xls[SCC_COL])

best_id, best_hits = None, -1
for c in [c for c in xls.columns if "no" in c.lower()]:
    hits = pd.to_numeric(xls[c], errors="coerce").isin(cow["stem"]).sum()
    if hits > best_hits: best_id, best_hits = c, hits
xls["_id"] = pd.to_numeric(xls[best_id], errors="coerce")
merged_oof = xls.merge(cow, left_on="_id", right_on="stem", how="inner")

enc = lambda s, k: s.astype(str).str.lower().str.contains(k).astype(int)
if CMT_COL: merged_oof["CMT_bin"] = enc(merged_oof[CMT_COL], "positive")
if ML_COL:  merged_oof["ML_bin"]  = enc(merged_oof[ML_COL],  "mastitis")

# 4) ===== EXPLICIT FLOW COUNT: why n is 85 for CMT but ~77 for SCC =====
print("\n" + "=" * 62)
print("SUBSET FLOW COUNT  (this is the 85 vs 77 explanation)")
print("=" * 62)
print(f"  Thermal images (pooled)            : {len(all_paths)}")
print(f"  Cows imaged                        : {len(set(all_groups))}")
print(f"  Cows matched to the blood/SCC sheet: {len(merged_oof)}")
if "CMT_bin" in merged_oof:
    n_cmt = merged_oof["CMT_bin"].notna().sum()
    print(f"  â†’ with a paired CMT record  (Table 9): n = {n_cmt}")
if "_SCC_num" in merged_oof:
    n_scc = merged_oof["_SCC_num"].notna().sum()
    n_drop = merged_oof["_SCC_num"].isna().sum()
    print(f"  â†’ with a NUMERIC SCC value  (Table 8): n = {n_scc}"
          f"   ({n_drop} dropped: SCC missing / non-numeric bound)")

# 5) leakage-free SCC significance (Mannâ€“Whitney) + agreement (Cohen's kappa)
d = merged_oof.dropna(subset=["_SCC_num"]).copy() if "_SCC_num" in merged_oof else merged_oof.iloc[0:0]
if len(d):
    scc = d["_SCC_num"].astype(float).values
    print("\nSCC as discriminator (leakage-free OOF DL label):")
    for lbl, col in [("vs CMT", "CMT_bin"), ("vs DL-OOF", "DL_bin"), ("vs old-ML", "ML_bin")]:
        if col in d:
            y = d[col].values
            if len(set(y)) == 2 and min((y == 0).sum(), (y == 1).sum()) >= 3:
                _, p = mannwhitneyu(scc[y == 1], scc[y == 0], alternative="greater")
                print(f"  [{lbl}] n={len(y)}  p={p:.4g}  SCC-AUC={roc_auc_score(y, scc):.3f}")

def _agree(ref, hat):
    tn, fp, fn, tp = confusion_matrix(ref, hat, labels=[0, 1]).ravel()
    return dict(n=len(ref), Accuracy=round(accuracy_score(ref, hat), 3),
                Sensitivity=round(tp/(tp+fn), 3) if tp+fn else np.nan,
                Specificity=round(tn/(tn+fp), 3) if tn+fp else np.nan,
                Kappa=round(cohen_kappa_score(ref, hat), 3))
print("\nAgreement (leakage-free OOF DL label):")
if "CMT_bin" in merged_oof:
    print("  DL-OOF vs CMT     :", _agree(merged_oof["CMT_bin"], merged_oof["DL_bin"]))
    if "ML_bin" in merged_oof:
        print("  old-ML vs CMT     :", _agree(merged_oof["CMT_bin"], merged_oof["ML_bin"]))
if "ML_bin" in merged_oof:
    print("  DL-OOF vs old-ML  :", _agree(merged_oof["ML_bin"], merged_oof["DL_bin"]))

_out = os.path.join(ROOT, "Mastitis_All_Comp", "merged_DL_OOF_SCC.xlsx")
merged_oof.to_excel(_out, index=False)
print(f"\nðŸ’¾ saved leakage-free merge: {_out}")
print("â†’ Use `merged_oof` (not the single-split merge) for Tables 8 and 9.")

# ==========================================================================
# CELL D (STANDALONE) â€” leakage-free OOF SCC/CMT merge + 85-vs-77 flow count
#   Rebuilds all_paths / folds with the SAME code+seed used to make the CV
#   pickles, then VERIFIES the reconstructed order matches the saved oof_true
#   before trusting the mapping. Runs on a fresh runtime.
# ==========================================================================
import os, re, glob, pickle, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix, accuracy_score

# ---- config (must match your Â§3/Â§4) ----
ROOT        = ROOT
OUT_DIR     = os.path.join(ROOT, "DL_results")
CLASS_NAMES = ["Healthy", "Mastitic"]
VALID_EXT   = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
SEED        = 42
XLSX_PATH   = os.path.join(ROOT, "Mastitis_All_Comp", "blood parameters& ML prediction.xlsx")
SHEET       = 0

os.makedirs(ROOT, exist_ok=True)

def cow_id_from_name(fname):
    return os.path.splitext(fname)[0].rstrip(".").strip()

# ---- load cv_results if needed ----
try:
    cv_results
except NameError:
    MZ = {"resnet50":"ResNet-50","densenet121":"DenseNet-121","efficientnet_b0":"EfficientNet-B0",
          "mobilenetv2_100":"MobileNetV2","inception_v3":"Inception-V3",
          "vit_base_patch16_224":"ViT-B/16","swin_tiny_patch4_window7_224":"Swin-Tiny"}
    cv_results = {}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "cv_result_*.pkl"))):
        t = os.path.basename(f).replace("cv_result_","").replace(".pkl","")
        with open(f,"rb") as fh: cv_results[MZ.get(t,t)] = pickle.load(fh)

best = max(cv_results, key=lambda n: cv_results[n]["agg"]["AUC"][0])
print("Using OOF predictions of:", best)

# ---- rebuild pooled dataset EXACTLY as Â§20 ----
all_paths, all_labels, all_groups = [], [], []
for lbl, cls in enumerate(CLASS_NAMES):
    d = os.path.join(ROOT, cls)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(VALID_EXT):
            all_paths.append(os.path.join(d, f))
            all_labels.append(lbl)
            all_groups.append(f"{cls}__{cow_id_from_name(f)}")
all_paths, all_labels, all_groups = map(np.array, (all_paths, all_labels, all_groups))
print(f"Pooled images: {len(all_paths)} | cows: {len(set(all_groups))} | {Counter(all_labels.tolist())}")

# ---- rebuild folds EXACTLY as Â§21 ----
sgkf  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(sgkf.split(all_paths, all_labels, groups=all_groups))
oof_order = np.concatenate([te for (_tr, te) in folds])

# ---- CRITICAL sanity check: reconstructed order must match saved oof_true ----
recon_true = all_labels[oof_order]
saved_true = np.asarray(cv_results[best]["oof_true"], int)
if len(recon_true) != len(saved_true) or not np.array_equal(recon_true, saved_true):
    raise RuntimeError(
        "âŒ Reconstructed OOF order does NOT match saved oof_true.\n"
        "   The dataset folder contents or SEED differ from when the CV was run.\n"
        "   Do NOT trust this mapping. Re-run Â§20â€“Â§21 in the same session instead.")
print("âœ… sanity check passed â€” OOF order matches saved labels exactly.")

oof_prob = np.asarray(cv_results[best]["oof_prob"], float)
oof_paths = all_paths[oof_order]

# ---- per-COW aggregation (mean OOF prob over that cow's images) ----
def _stem(cow_key):                        # cow_key like 'Healthy__57' -> 57
    d = re.findall(r"\d+", cow_key.split("__")[-1])
    return int(d[0]) if d else None
recs = pd.DataFrame({"cow": all_groups[oof_order], "prob": oof_prob})
cow = (recs.groupby("cow").agg(DL_conf=("prob","mean"), n_img=("prob","size")).reset_index())
cow["stem"]   = cow["cow"].map(_stem)
cow = cow.dropna(subset=["stem"]); cow["stem"] = cow["stem"].astype(int)
cow["DL_pred"] = np.where(cow["DL_conf"] >= 0.5, "Mastitis", "Healthy")
cow["DL_bin"]  = (cow["DL_pred"] == "Mastitis").astype(int)
print(f"Cows with OOF predictions: {len(cow)}")

# ---- merge with blood/SCC sheet ----
xls = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
xls.columns = [str(c).strip() for c in xls.columns]
def clean_scc(s):
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(s.replace({"<90000":"45000",">9000000":"9000000","":np.nan,"nan":np.nan}),
                         errors="coerce")
SCC_COL = next((c for c in xls.columns if "cell count" in c.lower() or c.lower()=="scc"), None)
CMT_COL = next((c for c in xls.columns if "california" in c.lower()), None)
ML_COL  = next((c for c in xls.columns if "prediction" in c.lower()), None)
if SCC_COL: xls["_SCC_num"] = clean_scc(xls[SCC_COL])
best_id, best_hits = None, -1
for c in [c for c in xls.columns if "no" in c.lower()]:
    h = pd.to_numeric(xls[c], errors="coerce").isin(cow["stem"]).sum()
    if h > best_hits: best_id, best_hits = c, h
xls["_id"] = pd.to_numeric(xls[best_id], errors="coerce")
merged_oof = xls.merge(cow, left_on="_id", right_on="stem", how="inner")
enc = lambda s,k: s.astype(str).str.lower().str.contains(k).astype(int)
if CMT_COL: merged_oof["CMT_bin"] = enc(merged_oof[CMT_COL], "positive")
if ML_COL:  merged_oof["ML_bin"]  = enc(merged_oof[ML_COL],  "mastitis")

# ---- 85-vs-77 FLOW COUNT ----
print("\n" + "="*60 + "\nSUBSET FLOW COUNT (the 85 vs 77 explanation)\n" + "="*60)
print(f"  Pooled images                       : {len(all_paths)}")
print(f"  Cows imaged                         : {len(set(all_groups))}")
print(f"  Cows matched to blood/SCC sheet     : {len(merged_oof)}")
if "CMT_bin" in merged_oof:
    print(f"  â†’ with paired CMT record  (Table 9) : n = {merged_oof['CMT_bin'].notna().sum()}")
if "_SCC_num" in merged_oof:
    n_scc  = merged_oof['_SCC_num'].notna().sum()
    n_drop = merged_oof['_SCC_num'].isna().sum()
    print(f"  â†’ with numeric SCC value  (Table 8) : n = {n_scc}  ({n_drop} dropped: missing/non-numeric)")

# ---- leakage-free SCC (Mannâ€“Whitney) + agreement (Cohen's kappa) ----
d = merged_oof.dropna(subset=["_SCC_num"]).copy() if "_SCC_num" in merged_oof else merged_oof.iloc[:0]
if len(d):
    scc = d["_SCC_num"].astype(float).values
    print("\nSCC discriminator (leakage-free OOF):")
    for lbl,col in [("vs CMT","CMT_bin"),("vs DL-OOF","DL_bin"),("vs old-ML","ML_bin")]:
        if col in d:
            y = d[col].values
            if len(set(y))==2 and min((y==0).sum(),(y==1).sum())>=3:
                _,p = mannwhitneyu(scc[y==1], scc[y==0], alternative="greater")
                print(f"  [{lbl}] n={len(y)}  p={p:.4g}  SCC-AUC={roc_auc_score(y,scc):.3f}")
def _ag(ref,hat):
    tn,fp,fn,tp = confusion_matrix(ref,hat,labels=[0,1]).ravel()
    return dict(n=len(ref), Acc=round(accuracy_score(ref,hat),3),
                Sens=round(tp/(tp+fn),3) if tp+fn else np.nan,
                Spec=round(tn/(tn+fp),3) if tn+fp else np.nan,
                Kappa=round(cohen_kappa_score(ref,hat),3))
print("\nAgreement (leakage-free OOF):")
if "CMT_bin" in merged_oof:
    print("  DL-OOF vs CMT    :", _ag(merged_oof["CMT_bin"], merged_oof["DL_bin"]))
    if "ML_bin" in merged_oof:
        print("  old-ML vs CMT    :", _ag(merged_oof["CMT_bin"], merged_oof["ML_bin"]))
if "ML_bin" in merged_oof:
    print("  DL-OOF vs old-ML :", _ag(merged_oof["ML_bin"], merged_oof["DL_bin"]))

merged_oof.to_excel(os.path.join(ROOT, "Mastitis_All_Comp", "merged_DL_OOF_SCC.xlsx"), index=False)
print("\nðŸ’¾ saved leakage-free merge â†’ use `merged_oof` for Tables 8 & 9.")

import os, pandas as pd
ROOT = ROOT
ids = []
for cls in ["Healthy", "Mastitis"]:
    for sub in ["train", "val", "test"]:
        d = os.path.join(ROOT, sub, cls)
        if os.path.isdir(d):
            for f in os.listdir(d):
                ids.append({"split": sub, "class": cls, "cow_id": os.path.splitext(f)[0]})
pd.DataFrame(ids).to_csv(os.path.join(ROOT, "train_val_test_cow_ids.csv"), index=False)
print("done:", len(ids))

import pickle, glob, os

OUT_DIR = os.path.join(ROOT, "DL_results")   # Ù„Ùˆ Ù…Ø®ØªÙ„Ù Ø¹Ù†Ø¯Ùƒ ØºÙŠØ±ÙŠÙ‡

for f in sorted(glob.glob(os.path.join(OUT_DIR, "cv_result_*.pkl"))):
    with open(f, "rb") as fh:
        r = pickle.load(fh)
    print(os.path.basename(f), "-> keys:", list(r.keys()))
    if "agg" in r:
        print("   agg keys:", list(r["agg"].keys()))
    break   # Ø¨Ø³ ÙˆØ§Ø­Ø¯ ÙƒÙØ§ÙŠØ© Ø¹Ø´Ø§Ù† Ù†Ø´ÙˆÙ Ø§Ù„Ø´ÙƒÙ„ Ø§Ù„Ø¹Ø§Ù…

import pickle, glob, os, numpy as np

OUT_DIR = os.path.join(ROOT, "DL_results")
MZ = {"resnet50":"ResNet-50","densenet121":"DenseNet-121","efficientnet_b0":"EfficientNet-B0",
      "mobilenetv2_100":"MobileNetV2","inception_v3":"Inception-V3",
      "vit_base_patch16_224":"ViT-B/16","swin_tiny_patch4_window7_224":"Swin-Tiny"}

def get_fold_values(per_fold, metric):
    """Handle both {metric: [v1..v5]} and [{metric: v}, {metric: v}, ...] shapes."""
    if isinstance(per_fold, dict):
        return np.asarray(per_fold[metric], dtype=float)
    elif isinstance(per_fold, list):
        return np.asarray([fold[metric] for fold in per_fold], dtype=float)
    else:
        raise TypeError(f"Unrecognized per_fold type: {type(per_fold)}")

results = {}
for tag, name in MZ.items():
    f = os.path.join(OUT_DIR, f"cv_result_{tag}.pkl")
    if not os.path.exists(f):
        continue
    with open(f, "rb") as fh:
        r = pickle.load(fh)
    results[name] = r["per_fold"]

# sanity check on one model first
print("Example structure (DenseNet-121):", type(results["DenseNet-121"]))
print(results["DenseNet-121"])
print()

for model in ["ViT-B/16", "Swin-Tiny"]:
    if model not in results:
        print(f"âš ï¸ {model} not found â€” check tag spelling in MZ")
        continue
    for metric in ["Sensitivity", "F1"]:
        vals = get_fold_values(results[model], metric)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        print(f"{model:12s} {metric:12s}: values={np.round(vals,3).tolist()}  "
              f"median={med:.3f}  IQR=[{q1:.3f}, {q3:.3f}]")

import os
d = os.path.join(ROOT, "Healthy")  # Ø£Ùˆ Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„ÙØ¹Ù„ÙŠ Ù„Ù…Ø¬Ù„Ø¯ Healthy
matches = [f for f in os.listdir(d) if f.startswith("25070343")]
print(matches)

print("ROOT =", ROOT)
d = os.path.join(ROOT, "Healthy")
print("Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„ÙƒØ§Ù…Ù„:", d)
files = sorted(os.listdir(d))
print("Ø¥Ø¬Ù…Ø§Ù„ÙŠ Ø§Ù„Ù…Ù„ÙØ§Øª:", len(files))
print("Ø£ÙˆÙ„ 20 Ø§Ø³Ù…:", files[:20])

import os, pickle, numpy as np, pandas as pd

OUT_DIR = os.path.join(ROOT, "DL_results")
MODEL_ZOO = {
    "ResNet-50": "resnet50", "DenseNet-121": "densenet121",
    "EfficientNet-B0": "efficientnet_b0", "MobileNetV2": "mobilenetv2_100",
    "Inception-V3": "inception_v3", "ViT-B/16": "vit_base_patch16_224",
    "Swin-Tiny": "swin_tiny_patch4_window7_224",
}

rows = []
found, missing = [], []

for name, timm_name in MODEL_ZOO.items():
    # single split
    p1 = os.path.join(OUT_DIR, f"result_{timm_name}.pkl")
    if os.path.exists(p1):
        d = pickle.load(open(p1, "rb"))
        for i, (yt, yp) in enumerate(zip(d["y_true"], d["y_prob"])):
            rows.append({"model": name, "split_type": "held_out_test", "fold": None,
                         "row": i, "y_true": int(yt), "y_prob": float(yp)})
        found.append(f"{name} (single split)")
    else:
        missing.append(f"{name} (single split)")

    # 5-fold CV
    p2 = os.path.join(OUT_DIR, f"cv_result_{timm_name}.pkl")
    if os.path.exists(p2):
        d = pickle.load(open(p2, "rb"))
        for i, (yt, yp) in enumerate(zip(d["oof_true"], d["oof_prob"])):
            rows.append({"model": name, "split_type": "cv_oof", "fold": None,
                         "row": i, "y_true": int(yt), "y_prob": float(yp)})
        found.append(f"{name} (5-fold CV)")
    else:
        missing.append(f"{name} (5-fold CV)")

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "ALL_per_image_predictions.csv")
df.to_csv(out_path, index=False)

print(f"âœ… Saved {len(df)} rows to {out_path}")
print(f"\nFOUND ({len(found)}):")
for x in found: print("  âœ”", x)
print(f"\nMISSING ({len(missing)}):")
for x in missing: print("  âœ˜", x)

import os

# --- image folders: adjust if your class-folder names differ ---
CLASS_FOLDERS = {
    "Healthy": 0,
    "Mastitis": 1,
}

# --- checkpoint of the model you already trained and are reporting on ---
BEST_MODEL_TIMM_NAME = "densenet121"          # matches MODEL_ZOO["DenseNet-121"]
BEST_MODEL_CKPT = os.path.join(OUT_DIR, f"{BEST_MODEL_TIMM_NAME}.pt")
ROI_OUT_DIR = os.path.join(ROOT, "DL_results_ROI")   # new outputs go here, originals untouched
os.makedirs(ROI_OUT_DIR, exist_ok=True)

# --- reproducibility: MUST match the RANDOM_STATE / seed used earlier in
#     this notebook so the held-out split is identical and the
#     ROI-vs-original comparison is apples-to-apples ---
RANDOM_STATE = 42   # <-- CHANGE THIS to whatever you actually used above

IMG_SIZE = 224

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# torch, torch.nn, torch.nn.functional, timm, DEVICE should already be
# defined earlier in this notebook -- if not, uncomment:
# import torch, torch.nn as nn, torch.nn.functional as F, timm
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_images(root, class_folders):
    rows = []
    for folder_name, label in class_folders.items():
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"WARNING: folder not found: {folder_path} -- fix CLASS_FOLDERS")
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                rows.append({"path": os.path.join(folder_path, fname), "label": label, "fname": fname})
    df = pd.DataFrame(rows)
    print(f"Collected {len(df)} images ({(df.label==0).sum()} healthy, {(df.label==1).sum()} mastitic)")
    return df

all_images_df = collect_images(ROOT, CLASS_FOLDERS)

def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def derive_udder_mask(bgr_img, margin_frac=0.08, min_area_frac=0.03):
    """
    Returns (mask, bbox). mask: uint8 0/255, same size as input.
    bbox: (x1, y1, x2, y2) with margin, clipped to image bounds.
    Returns (None, None) if no sufficiently large warm blob was found.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n_labels <= 1:
        return None, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return None, None
    largest_idx = 1 + int(np.argmax(areas))
    area = stats[largest_idx, cv2.CC_STAT_AREA]

    h, w = gray.shape
    if area < min_area_frac * h * w:
        return None, None

    mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    x, y, bw, bh = (stats[largest_idx, cv2.CC_STAT_LEFT], stats[largest_idx, cv2.CC_STAT_TOP],
                     stats[largest_idx, cv2.CC_STAT_WIDTH], stats[largest_idx, cv2.CC_STAT_HEIGHT])
    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)

    return mask, (x1, y1, x2, y2)


def sanity_check_masks(df, n=8, seed=0):
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for i, (_, row) in enumerate(sample.iterrows()):
        bgr = load_bgr(row["path"])
        mask, bbox = derive_udder_mask(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"{row['fname'][:14]}\n({'H' if row['label']==0 else 'M'})", fontsize=8)
        axes[0, i].axis("off")
        overlay = rgb.copy()
        if mask is not None:
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("mask (red) + crop bbox (green)" if mask is not None else "NO MASK FOUND", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(ROI_OUT_DIR, "mask_sanity_check.png"), dpi=120)
    plt.show()
    print("LOOK AT THIS before trusting anything downstream.")

sanity_check_masks(all_images_df, n=8)

def load_trained_model(timm_name, ckpt_path, num_classes=2):
    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def find_last_conv_layer(model):
    last_conv, last_name = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv, last_name = module, name
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found -- check architecture.")
    print(f"Grad-CAM target layer auto-detected: {last_name}")
    return last_conv


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, torch.softmax(output, dim=1)[0].detach().cpu().numpy()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def preprocess_for_model(bgr_img, size=IMG_SIZE):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return x.to(DEVICE)


best_model = load_trained_model(BEST_MODEL_TIMM_NAME, BEST_MODEL_CKPT)
target_layer = find_last_conv_layer(best_model)
gradcam = GradCAM(best_model, target_layer)

def peak_outside_mask_fraction(cam, mask, size, top_frac=0.20):
    h, w = size
    cam_resized = cv2.resize(cam, (w, h))
    flat = cam_resized.flatten()
    n_top = max(1, int(top_frac * flat.size))
    thresh = np.partition(flat, -n_top)[-n_top]
    peak_binary = (cam_resized >= thresh)
    mask_binary = (mask > 0)
    outside = peak_binary & (~mask_binary)
    return outside.sum() / max(1, peak_binary.sum())


def run_saliency_vs_mask_analysis(df, model, gradcam_obj, top_frac=0.20):
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        bgr = load_bgr(row["path"])
        h, w = bgr.shape[:2]
        mask, bbox = derive_udder_mask(bgr)
        if mask is None:
            skipped += 1
            continue
        x = preprocess_for_model(bgr)
        cam, pred_class, probs = gradcam_obj(x, class_idx=None)
        frac_outside = peak_outside_mask_fraction(cam, mask, (h, w), top_frac=top_frac)
        rows.append({
            "fname": row["fname"], "true_label": row["label"], "pred_label": pred_class,
            "correct": int(row["label"] == pred_class), "prob_mastitic": float(probs[1]),
            "frac_peak_activation_outside_udder_mask": frac_outside,
        })
    result_df = pd.DataFrame(rows)
    print(f"Analyzed {len(result_df)} images, skipped {skipped} (no reliable mask found)")
    return result_df


saliency_df = run_saliency_vs_mask_analysis(all_images_df, best_model, gradcam, top_frac=0.20)
saliency_df.to_csv(os.path.join(ROI_OUT_DIR, "saliency_vs_udder_mask.csv"), index=False)

summary = {
    "n_images": len(saliency_df),
    "mean_frac_outside": saliency_df["frac_peak_activation_outside_udder_mask"].mean(),
    "median_frac_outside": saliency_df["frac_peak_activation_outside_udder_mask"].median(),
    "mean_frac_outside_correct": saliency_df.loc[saliency_df.correct == 1, "frac_peak_activation_outside_udder_mask"].mean(),
    "mean_frac_outside_incorrect": saliency_df.loc[saliency_df.correct == 0, "frac_peak_activation_outside_udder_mask"].mean(),
}
print(summary)
pd.Series(summary).to_csv(os.path.join(ROI_OUT_DIR, "saliency_vs_udder_mask_SUMMARY.csv"))

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def build_roi_cropped_dataset(df, out_dir, size=IMG_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        bgr = load_bgr(row["path"])
        h, w = bgr.shape[:2]
        mask, bbox = derive_udder_mask(bgr)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = bgr[y1:y2, x1:x2]
            fallback = False
        else:
            cx, cy = w // 2, h // 2
            half_w, half_h = int(w * 0.35), int(h * 0.35)
            crop = bgr[max(0, cy - half_h):cy + half_h, max(0, cx - half_w):cx + half_w]
            fallback = True
        crop = cv2.resize(crop, (size, size))
        out_path = os.path.join(out_dir, row["fname"])
        cv2.imwrite(out_path, crop)
        rows.append({**row.to_dict(), "roi_path": out_path, "used_fallback_crop": fallback})
    out_df = pd.DataFrame(rows)
    print(f"Built ROI-cropped dataset: {len(out_df)} images, {out_df['used_fallback_crop'].sum()} used fallback crop")
    return out_df


roi_df = build_roi_cropped_dataset(all_images_df, os.path.join(ROI_OUT_DIR, "cropped_images"))
roi_df.to_csv(os.path.join(ROI_OUT_DIR, "roi_dataset_manifest.csv"), index=False)


class ROIImageDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row["roi_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            rgb = np.fliplr(rgb).copy()
        x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


def train_densenet_on_roi(roi_df, random_state=RANDOM_STATE, epochs=20, lr=3e-4, weight_decay=1e-4, batch_size=32):
    train_df, test_df = train_test_split(roi_df, test_size=0.15, stratify=roi_df["label"], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=random_state)
    print(f"ROI split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_loader = DataLoader(ROIImageDataset(train_df, augment=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ROIImageDataset(val_df, augment=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ROIImageDataset(test_df, augment=False), batch_size=batch_size, shuffle=False)

    model = timm.create_model(BEST_MODEL_TIMM_NAME, pretrained=True, num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_auc, best_state = -1, None
    from sklearn.metrics import roc_auc_score
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                val_probs.extend(out.tolist()); val_true.extend(y.numpy().tolist())
        val_auc = roc_auc_score(val_true, val_probs)
        print(f"epoch {epoch+1}/{epochs}  val_AUC={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(ROI_OUT_DIR, f"{BEST_MODEL_TIMM_NAME}_ROI.pt"))

    model.eval()
    test_probs, test_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = torch.softmax(model(x.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(out.tolist()); test_true.extend(y.numpy().tolist())

    test_probs, test_true = np.array(test_probs), np.array(test_true)
    test_pred = (test_probs >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(test_true, test_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(test_true, test_pred), "auc": roc_auc_score(test_true, test_probs),
        "sensitivity": recall_score(test_true, test_pred), "specificity": tn / (tn + fp),
        "precision_ppv": precision_score(test_true, test_pred), "npv": tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
        "f1": f1_score(test_true, test_pred), "n_test": len(test_true),
    }
    print("ROI-cropped DenseNet-121 held-out test metrics:", metrics)
    pd.Series(metrics).to_csv(os.path.join(ROI_OUT_DIR, "densenet121_ROI_test_metrics.csv"))
    return metrics, model


# Ø´ØºÙ‘Ù„ÙŠÙ‡Ø§ Ù„Ù…Ø§ ØªÙƒÙˆÙ†ÙŠ Ø¬Ø§Ù‡Ø²Ø© (GPU ÙŠÙØ¶Ù„ ÙŠÙƒÙˆÙ† Ø´ØºØ§Ù„):
# roi_metrics, roi_model = train_densenet_on_roi(roi_df)

import time

def load_bgr(path, retries=5, delay=1.5):
    """Robust read: retries on transient Google Drive I/O errors instead
    of crashing the whole loop. Returns None (and prints a warning) if the
    file truly can't be read after retries -- callers must skip None."""
    last_err = None
    for attempt in range(retries):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except OSError as e:
            last_err = e
            print(f"  [retry {attempt+1}/{retries}] Drive I/O hiccup on {os.path.basename(path)}: {e}")
            time.sleep(delay)
    print(f"  [SKIPPED] could not read after {retries} retries: {path}")
    return None




