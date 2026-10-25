import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import build_model, count_params

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LR = {"scratch": 1e-3, "feature_extract": 1e-3, "finetune": 3e-4}


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over the loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--mode", default="finetune")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None, help="for quick smoke tests")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    name = args.name or f"{args.arch}_{args.mode}"
    lr = args.lr or DEFAULT_LR[args.mode]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_loader, val_loader, _, classes = get_dataloaders(args.batch_size, args.num_workers)
    model = build_model(args.arch, args.mode, num_classes=len(classes)).to(device)
    total, trainable = count_params(model)
    print(f"{name} | device={device} | params total={total:,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_val_acc = [], 0.0
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if args.mode == "feature_extract":
            # frozen layers should not update BatchNorm statistics either
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        model.eval()
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None, args.max_batches)
        scheduler.step()

        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=time.time() - t0))
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                        "mode": args.mode, "classes": classes}, ROOT / "models" / f"{name}.pt")

    with open(ROOT / "reports" / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val acc: {best_val_acc:.3f} -> models/{name}.pt")


if __name__ == "__main__":
    main()