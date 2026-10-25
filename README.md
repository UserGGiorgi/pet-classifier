# Pet Breed Classifier

Classifies photos of cats and dogs into 37 breeds (Oxford-IIIT Pet Dataset) using transfer learning with ResNet18. The final model is exported to ONNX and INT8-quantized, and served through a FastAPI app with a small web page.

## Results

For full details, see [REPORT.md](REPORT.md).

| Model (final) | Test accuracy | Top-5 | Macro F1 |
|---|---|---|---|
| ResNet18, feature extraction | 89.59% | 99.24% | 0.894 |

### Deployment variants
| Variant | Size | Latency (forward pass, i7-12700) | Test accuracy |
|---|---|---|---|
| PyTorch FP32 | 44.9 MB | 13.6 ms | 89.59% |
| ONNX FP32 | 44.8 MB | 7.1 ms | 89.59% |
| ONNX INT8 | 11.3 MB | 2.7 ms | 89.21% |

## Project structure

```
api/             FastAPI service (main.py) and web page (static/index.html)
artifacts/       Deployable INT8 model and class names
notebooks/       Data exploration, experiment comparison, error analysis
reports/         Training histories, test metrics, confusion matrices, optimization table
src/             data.py, model.py, train.py, evaluate.py, optimize.py, export_artifacts.py
```

## Run the demo (no training needed)

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
uvicorn api.main:app
```

Open http://127.0.0.1:8000 for the web page or http://127.0.0.1:8000/docs for the API docs.

Endpoints: `GET /health`, `POST /classify` (multipart field `file`, optional `top_k`).

```bash
curl -F "file=@pet.jpg" http://127.0.0.1:8000/classify
```

## Reproduce training and evaluation

Full dependencies (PyTorch etc.): `pip install -r requirements.txt`. Training was done on a free Colab T4 GPU.

```bash
python -m src.train --arch resnet18 --mode scratch --epochs 10
python -m src.train --arch resnet18 --mode feature_extract --epochs 10
python -m src.train --arch resnet18 --mode finetune --epochs 10
python -m src.train --arch mobilenet_v3_small --mode finetune --epochs 10
```

Each run saves `models/<name>.pt` and `reports/<name>_history.json`. The dataset downloads automatically to `data/` (about 800 MB). Weights in `models/` are not committed.

```bash
python -m src.evaluate --name resnet18_feature_extract     # test metrics, confusion matrix
python -m src.optimize --names resnet18_feature_extract mobilenet_v3_small_finetune
python -m src.export_artifacts                             # regenerates artifacts/
python -m src.check_preprocess                             # train/serve preprocessing check
```

## Known limitations

- Closed-set classifier: a non-pet image still gets a breed (a car photo produced top-1 confidence of about 12%, spread across three breeds).
- Most errors are between visually similar breeds (Staffordshire vs American Pit Bull Terrier, Ragdoll vs Birman, Egyptian Mau vs Bengal).
- MobileNetV3-small INT8 collapsed to about 3% accuracy with default static quantization, so it is not deployed.
- Single training seed per experiment.