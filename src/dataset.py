"""Dataset and data-loader helpers for image-folder plant disease datasets."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_datasets(config: dict) -> dict[str, datasets.ImageFolder]:
    """Create datasets for the configured train/validation/test directories."""
    image_size = config["data"]["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    result = {}
    for split in ("train", "val", "test"):
        directory = Path(config["data"][f"{split}_dir"])
        if directory.is_dir():
            result[split] = datasets.ImageFolder(directory, transform=transform)
    if "train" not in result:
        raise FileNotFoundError(
            f"Training data directory does not exist: {config['data']['train_dir']}"
        )
    return result


def create_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Create data loaders for each available dataset split."""
    datasets_by_split = create_datasets(config)
    return {
        split: DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["data"]["num_workers"],
        )
        for split, dataset in datasets_by_split.items()
    }
