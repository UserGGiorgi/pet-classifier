import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

from src.data import get_dataloaders, get_datasets
from src.evaluate import load_model

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models" / "onnx"


class PetCalibration(CalibrationDataReader):
    """Feeds validation images (no labels) so quantization can measure value ranges."""
    def __init__(self, n_images=128, batch=16):
        _, val, _, _ = get_datasets()
        xs = np.stack([val[i][0].numpy() for i in range(n_images)])
        self.batches = iter([{"input": xs[i:i + batch]} for i in range(0, n_images, batch)])

    def get_next(self):
        return next(self.batches, None)


def export_onnx(name):
    model, _ = load_model(name, torch.device("cpu"))
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    path = ONNX_DIR / f"{name}_fp32.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False)
    return model, path


def quantize(fp32_path, int8_path):
    quantize_static(str(fp32_path), str(int8_path), PetCalibration(),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timeit(fn, runs=100, warmup=10):
    for _ in range(warmup):
        fn()
    ms = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1000)
    return float(np.median(ms)), float(np.percentile(ms, 95))


@torch.no_grad()
def accuracy_all(runners, loader, max_batches=None):
    """One pass over the test set, feeding every variant the same batches."""
    correct = {k: 0 for k in runners}
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        xn, yn = x.numpy(), y.numpy()
        for k, f in runners.items():
            correct[k] += int((f(xn).argmax(1) == yn).sum())
        n += len(yn)
    return {k: v / n for k, v in correct.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None, help="limit test batches (quick run)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    _, _, test_loader, _ = get_dataloaders(64, num_workers=0)
    x1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
    rows = []

    for name in args.names:
        print(f"\n=== {name} ===")
        model, fp32_path = export_onnx(name)
        int8_path = ONNX_DIR / f"{name}_int8.onnx"
        quantize(fp32_path, int8_path)

        s32, s8 = make_session(fp32_path, args.threads), make_session(int8_path, args.threads)

        # sanity: exported model must match PyTorch
        with torch.no_grad():
            diff = np.abs(model(torch.from_numpy(x1)).numpy() - s32.run(None, {"input": x1})[0]).max()
        print(f"max |PyTorch - ONNX FP32| logit difference: {diff:.2e}")

        def torch_fn(xn):
            return model(torch.from_numpy(xn)).numpy()

        runners = {
            "PyTorch FP32": torch_fn,
            "ONNX FP32": lambda xn: s32.run(None, {"input": xn})[0],
            "ONNX INT8": lambda xn: s8.run(None, {"input": xn})[0],
        }
        sizes = {"PyTorch FP32": (ROOT / "models" / f"{name}.pt").stat().st_size,
                 "ONNX FP32": fp32_path.stat().st_size, "ONNX INT8": int8_path.stat().st_size}
        lat = {
            "PyTorch FP32": timeit(lambda: torch.no_grad()(model)(torch.from_numpy(x1))),
            "ONNX FP32": timeit(lambda: s32.run(None, {"input": x1})),
            "ONNX INT8": timeit(lambda: s8.run(None, {"input": x1})),
        }
        acc = accuracy_all(runners, test_loader, args.max_batches)

        for k in runners:
            rows.append(dict(model=name, variant=k, size_mb=round(sizes[k] / 1e6, 1),
                             latency_ms_median=round(lat[k][0], 1), latency_ms_p95=round(lat[k][1], 1),
                             test_acc=round(acc[k], 4)))

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "optimization.csv", index=False)


if __name__ == "__main__":
    main()