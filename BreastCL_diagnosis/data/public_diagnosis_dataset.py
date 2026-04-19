from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.augmentation import MultimodalInserter_New


class PublicDiagnosisDataset(Dataset):


    REQUIRED_COLUMNS = ("T2", "DWI", "SUB_concate")

    def __init__(
        self,
        csv_path: str | Path,
        data_root: Optional[str | Path] = None,
        label_column: str = "malignant",
        transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root) if data_root else None
        self.label_column = label_column
        self.df = pd.read_csv(self.csv_path)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing}")

        self.subjects = (
            self.df["Subject"].astype(str).tolist()
            if "Subject" in self.df.columns
            else [f"case_{i:04d}" for i in range(len(self.df))]
        )
        self.has_labels = self.label_column in self.df.columns
        self.labels = (
            torch.as_tensor(self.df[self.label_column].fillna(-1).astype(int).to_numpy(), dtype=torch.long)
            if self.has_labels
            else torch.full((len(self.df),), -1, dtype=torch.long)
        )

        self.t2_paths = [self._resolve_path(p) for p in self.df["T2"].tolist()]
        self.dwi_paths = [self._resolve_path(p) for p in self.df["DWI"].tolist()]
        self.sub_paths = [self._resolve_path(p) for p in self.df["SUB_concate"].tolist()]

        self.transform = transform or T.Compose(
            [
                MultimodalInserter_New(
                    dce_size=(336, 224, 128),
                    dwi_size=(256, 128, 32),
                    t2_size=(336, 224, 48),
                    rand=False,
                )
            ]
        )

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute() or self.data_root is None:
            return path
        return self.data_root / path

    @staticmethod
    def _load_volume(path: Path, add_channel: bool = True) -> np.ndarray:
        arr = np.load(path)
        if add_channel and arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        return arr

    @staticmethod
    def _prepare_dce(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        if arr.shape[0] != 6:
            arr = np.concatenate([arr] * 6, axis=0)
        return arr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        t2 = self._load_volume(self.t2_paths[index], add_channel=True)
        dwi = self._load_volume(self.dwi_paths[index], add_channel=True)
        sub = self._prepare_dce(self._load_volume(self.sub_paths[index], add_channel=False))

        sample = {"t2": t2, "dwi": dwi, "dce": sub}
        sample = self.transform(sample)

        return {
            "index": index,
            "subject": self.subjects[index],
            "t2": sample["t2"],
            "dwi": sample["dwi"],
            "sub": sample["dce"],
            "label": self.labels[index],
        }
