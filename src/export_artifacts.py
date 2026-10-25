import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
NAME = "resnet18_feature_extract"

ckpt = torch.load(ROOT / "models" / f"{NAME}.pt", map_location="cpu")
art = ROOT / "artifacts"
art.mkdir(exist_ok=True)

# class names as plain JSON, so the API doesn't need PyTorch to read them
(art / "classes.json").write_text(json.dumps(ckpt["classes"], indent=2), encoding="utf-8")
shutil.copy(ROOT / "models" / "onnx" / f"{NAME}_int8.onnx", art / "pet_resnet18_int8.onnx")
print("saved:", sorted(p.name for p in art.iterdir()))