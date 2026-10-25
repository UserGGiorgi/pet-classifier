import numpy as np
from torchvision.datasets import OxfordIIITPet

from api.main import preprocess
from src.data import DATA_DIR, get_transforms

raw = OxfordIIITPet(DATA_DIR, split="test", target_types="category")
tf = get_transforms(train=False)
for i in range(0, 100, 20):
    img, _ = raw[i]
    diff = np.abs(tf(img).numpy()[None] - preprocess(img)).max()
    print(f"image {i}: max abs difference {diff:.2e}")