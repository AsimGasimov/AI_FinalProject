"""Training CLI for SimpleCNN and EfficientNet-B0.

Usage:
    python -m src.cnn.train --model simple --epochs 15 --bs 64
    python -m src.cnn.train --model effnet --epochs 10 --bs 32 --mixed-precision
    python -m src.cnn.train --model effnet --smoke   # 1 epoch on ~200 images

AdamW + label smoothing 0.1 + cosine schedule, early stopping on val macro-F1
(patience 3). Best checkpoint -> models/{model}_best.pt with metadata.
Per-epoch history -> reports/history_{model}.json.
EfficientNet trains in two phases: frozen backbone (3 epochs, lr) then full
fine-tune (remaining epochs, lr/10).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CLASSES, SEED, settings
from src.cnn.dataset import get_dataloaders
from src.cnn.models import build_model, count_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def set_seed(seed: int = SEED) -> None:
    """Fix all RNGs for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: str,
                   ) -> tuple[float, float, float]:
    """Return (loss, top1_acc, macro_f1) on a loader."""
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(crit(out, y).item())
        preds += out.argmax(1).cpu().tolist()
        targets += y.cpu().tolist()
    acc = sum(p == t for p, t in zip(preds, targets)) / max(len(targets), 1)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return sum(losses) / max(len(losses), 1), acc, float(f1)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                    crit: nn.Module, device: str, scaler: torch.amp.GradScaler,
                    use_amp: bool, desc: str) -> tuple[float, float]:
    """One training epoch; returns (mean_loss, top1_acc)."""
    model.train()
    losses, correct, seen = [], 0, 0
    bar = tqdm(loader, desc=desc, ncols=100)
    for x, y in bar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        correct += (out.argmax(1) == y).sum().item()
        seen += y.numel()
        bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / seen:.3f}")
    return sum(losses) / max(len(losses), 1), correct / max(seen, 1)


def run_training(model_name: str, epochs: int, bs: int, lr: float,
                 mixed_precision: bool, limit: int | None,
                 num_workers: int | None, smoke: bool = False) -> dict:
    """Full training loop with phases, early stopping, checkpointing.

    Smoke runs (``smoke=True``) save to ``{model}_smoke.pt`` so a quick ETA
    measurement can never clobber a real ``{model}_best.pt`` checkpoint.
    """
    set_seed()
    device = settings.resolve_device()
    use_amp = mixed_precision and device == "cuda"
    log.info("model=%s device=%s amp=%s limit=%s", model_name, device, use_amp, limit)

    train_loader, val_loader, _ = get_dataloaders(bs, num_workers=num_workers, limit=limit)
    model = build_model(model_name).to(device)
    log.info("params=%s", f"{count_params(model):,}")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Phase plan: effnet = 3 frozen-head epochs @ lr, then full @ lr/10.
    if model_name == "effnet" and epochs > 3:
        phases = [("head", min(3, epochs), lr), ("full", epochs - 3, lr / 10)]
    else:
        phases = [("full", epochs, lr)]

    history: list[dict] = []
    best_f1, patience, bad_epochs, epoch_idx = -1.0, 3, 0, 0
    # Smoke runs must never clobber the real best checkpoint.
    ckpt_path = settings.models_dir / f"{model_name}_{'smoke' if smoke else 'best'}.pt"
    t0 = time.time()

    for phase_name, phase_epochs, phase_lr in phases:
        if model_name == "effnet":
            freeze = phase_name == "head"
            for p in model.features.parameters():  # type: ignore[union-attr]
                p.requires_grad = not freeze
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=phase_lr,
            weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(phase_epochs, 1))

        for _ in range(phase_epochs):
            epoch_idx += 1
            desc = f"[{model_name}/{phase_name}] epoch {epoch_idx}/{epochs}"
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit,
                                              device, scaler, use_amp, desc)
            va_loss, va_acc, va_f1 = evaluate_epoch(model, val_loader, device)
            sched.step()
            row = {"epoch": epoch_idx, "phase": phase_name, "train_loss": tr_loss,
                   "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                   "val_macro_f1": va_f1, "lr": phase_lr,
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            log.info("epoch %d: train_acc=%.3f val_acc=%.3f val_f1=%.3f",
                     epoch_idx, tr_acc, va_acc, va_f1)

            if va_f1 > best_f1:
                best_f1, bad_epochs = va_f1, 0
                torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                            "classes": CLASSES, "img_size": 224, "seed": SEED,
                            "val_macro_f1": va_f1, "val_acc": va_acc,
                            "epoch": epoch_idx, "params": count_params(model)},
                           ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.info("Early stopping at epoch %d (patience %d)", epoch_idx, patience)
                    _write_history(model_name, history)
                    return {"best_val_f1": best_f1, "epochs_run": epoch_idx,
                            "elapsed_s": time.time() - t0, "ckpt": str(ckpt_path)}

    _write_history(model_name, history)
    return {"best_val_f1": best_f1, "epochs_run": epoch_idx,
            "elapsed_s": time.time() - t0, "ckpt": str(ckpt_path)}


def _write_history(model_name: str, history: list[dict]) -> None:
    out = settings.reports_dir / f"history_{model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train FoodLens CNNs")
    ap.add_argument("--model", choices=["simple", "effnet"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on ~200 images (ETA measurement)")
    args = ap.parse_args()

    limit = 200 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    res = run_training(args.model, epochs, args.bs, args.lr,
                       args.mixed_precision, limit, args.num_workers,
                       smoke=args.smoke)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
