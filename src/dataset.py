"""PyTorch dataset for the five-class PlantVillage / PlantDoc experiments.

Reads CSV split manifests produced by src/data_prep.py. Images are loaded
from Hugging Face datasets on demand, not copied into the Git repository.

Usage:
    from src.dataset import PlantDiseaseDataset
    from torchvision import transforms
    ds = PlantDiseaseDataset(
        "data/splits/lab_train.csv",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    )
    image, label = ds[0]
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_SOURCES = {
    "plantvillage": ("geraldmc/plantvillage-full", None),
    "plantdoc": ("geraldmc/plantdoc-full", "v0.1.0"),
}

REQUIRED_COLUMNS = {
    "source", "row_id", "original_class", "class_name", "label_id"
}


class PlantDiseaseDataset(Dataset):
    """Return (RGB image tensor, class ID) from a split manifest."""

    def __init__(self, manifest_path, transform=None, revisions=None):
        self.manifest_path = Path(manifest_path)
        self.metadata = pd.read_csv(self.manifest_path)

        missing = REQUIRED_COLUMNS - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        if self.metadata.empty:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        if self.metadata[["source", "row_id", "original_class", "label_id"]].isna().any().any():
            raise ValueError("Required manifest fields contain missing values")
        if not self.metadata["label_id"].between(0, 4).all():
            raise ValueError("Expected label IDs between 0 and 4")
        if self.metadata.duplicated(subset=["source", "row_id"]).any():
            raise ValueError("Duplicate source + row_id in manifest")

        # Temporary smoke-test transform; replace at training time with the
        # preprocessing required by the selected pretrained model.
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        revisions = revisions or {}
        self.source_datasets = {}
        for source in self.metadata["source"].unique():
            if source not in DATASET_SOURCES:
                raise ValueError(f"Unknown data source: {source}")
            repository, default_revision = DATASET_SOURCES[source]
            revision = revisions.get(source, default_revision)
            self.source_datasets[source] = load_dataset(
                repository, split="train", revision=revision
            )
            max_idx = self.metadata.loc[
                self.metadata["source"] == source, "row_id"
            ].max()
            if max_idx >= len(self.source_datasets[source]):
                raise ValueError(f"Invalid row_id for {source}: {max_idx}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        sample = self.source_datasets[row["source"]][int(row["row_id"])]

        # Detect manifest / upstream dataset reordering before training.
        if sample["class_label"] != row["original_class"]:
            raise ValueError(
                f"Label mismatch at index {index}. "
                "Check the pinned dataset revision and original row order."
            )

        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image).__name__}")
        image = image.convert("RGB")

        return self.transform(image), int(row["label_id"])
