import os
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class FilenameDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        self.files = list(self.images_dir.glob("*.jpg"))
        if not self.files:
            raise RuntimeError(f"No JPG images found in {images_dir}")

        self.labels = sorted({f.stem.split("_")[0] for f in self.files})

        self.class_to_idx = {c: i for i, c in enumerate(self.labels)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = self.files[idx]

        label_str = img_path.stem.split("_")[0]
        label = self.class_to_idx[label_str]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, val_tf

def build_dataloaders(
    images_dir: str,
    batch_size: int,
    seed: int = 42,
    val_split: float = 0.2,
    num_workers: int = 4,
):
    train_tf, val_tf = get_transforms()

    full_dataset = FilenameDataset(
        images_dir=images_dir,
        transform=None
    )

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=generator
    )

    train_subset.dataset.transform = train_tf
    val_subset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, full_dataset.labels
