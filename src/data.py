from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_transforms(train: bool):
    """Training gets random augmentation; val/test get deterministic preprocessing."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_datasets(val_fraction: float = 0.15, seed: int = 42):
    """Returns train, val, test datasets.

    The dataset ships with an official trainval/test split. We keep the
    official test set untouched and carve a validation set out of trainval.
    """
    # Two copies of trainval: one with augmentation, one without,
    # so validation images are not randomly augmented.
    trainval_aug = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                                 transform=get_transforms(train=True), download=True)
    trainval_plain = OxfordIIITPet(DATA_DIR, split="trainval", target_types="category",
                                   transform=get_transforms(train=False), download=True)
    test = OxfordIIITPet(DATA_DIR, split="test", target_types="category",
                         transform=get_transforms(train=False), download=True)

    n = len(trainval_aug)
    n_val = int(n * val_fraction)
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n), [n - n_val, n_val], generator=generator)

    train = torch.utils.data.Subset(trainval_aug, train_idx.indices)
    val = torch.utils.data.Subset(trainval_plain, val_idx.indices)
    return train, val, test, trainval_aug.classes


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    train, val, test, classes = get_datasets()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, classes