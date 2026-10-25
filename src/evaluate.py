import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data import get_dataloaders
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent


def load_model(name, device):
    ckpt = torch.load(ROOT / "models" / f"{name}.pt", map_location=device)
    # build_model may download pretrained weights first; they are overwritten below
    model = build_model(ckpt["arch"], ckpt["mode"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


@torch.no_grad()
def predict(model, loader, device):
    probs, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="checkpoint name, e.g. resnet18_feature_extract")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.name, device)
    _, _, test_loader, _ = get_dataloaders(args.batch_size, num_workers=0)

    probs, labels = predict(model, test_loader, device)
    preds = probs.argmax(1)

    acc = float((preds == labels).mean())
    top5 = float(np.mean([l in np.argsort(p_)[-5:] for p_, l in zip(probs, labels)]))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    report = classification_report(labels, preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"{args.name} on TEST ({len(labels)} images)")
    print(f"accuracy {acc:.4f} | top-5 {top5:.4f} | macro F1 {macro_f1:.4f}")

    per_class = sorted(((c, report[c]["f1-score"], report[c]["precision"], report[c]["recall"])
                        for c in classes), key=lambda t: t[1])
    print("\nWorst 5 breeds by F1:")
    for c, f1, pr, rc in per_class[:5]:
        print(f"  {c:28s} F1 {f1:.2f}  precision {pr:.2f}  recall {rc:.2f}")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("\nMost common confusions (true -> predicted):")
    for idx in np.argsort(off.ravel())[::-1][:10]:
        t, pr = np.unravel_index(idx, off.shape)
        print(f"  {classes[t]} -> {classes[pr]}: {off[t, pr]}")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / f"{args.name}_test_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "top5": top5, "macro_f1": macro_f1, "per_class": report}, f, indent=2)
    np.savez(reports / f"{args.name}_test_preds.npz", probs=probs, labels=labels)

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix: {args.name}")
    plt.tight_layout()
    plt.savefig(reports / f"{args.name}_confusion.png", dpi=150)


if __name__ == "__main__":
    main()