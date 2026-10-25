import torch
import torchvision

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA GPU available:", torch.cuda.is_available())

# Tiny sanity test: random "image batch" through the shape a model expects
x = torch.randn(4, 3, 224, 224)  # 4 images, 3 color channels, 224x224 pixels
print("Batch shape:", x.shape)