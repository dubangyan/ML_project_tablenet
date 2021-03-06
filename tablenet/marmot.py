#Marmot 数据集处理

import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import List
from albumentations import Compose
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MarmotDataset(Dataset):

    def __init__(self, data: List[Path], transforms: Compose = None) -> None:
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample_id = self.data[item].stem

        image_path = self.data[item]
        table_path = self.data[item].parent.parent.joinpath("table_mask", sample_id + ".bmp")
        column_path = self.data[item].parent.parent.joinpath("column_mask", sample_id + ".bmp")

        image = np.array(Image.open(image_path))
        table_mask = np.expand_dims(np.array(Image.open(table_path)), axis=2)
        column_mask = np.expand_dims(np.array(Image.open(column_path)), axis=2)
        mask = np.concatenate([table_mask, column_mask], axis=2) / 255
        sample = {"image": image, "mask": mask}
        if self.transforms:
            sample = self.transforms(image=image, mask=mask)

        image = sample["image"]
        mask_table = sample["mask"][:, :, 0].unsqueeze(0)
        mask_column = sample["mask"][:, :, 1].unsqueeze(0)
        return image, mask_table, mask_column


class MarmotDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./data", transforms_preprocessing: Compose = None,
                 transforms_augmentation: Compose = None, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.data = list(Path(data_dir).rglob("*.bmp"))
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        n_samples = len(self.data)
        self.data.sort()
        train_slice = slice(0, int(n_samples * 0.8))
        val_slice = slice(int(n_samples * 0.8), int(n_samples * 0.9))
        test_slice = slice(int(n_samples * 0.9), n_samples)

        self.complaint_train = MarmotDataset(self.data[train_slice], transforms=self.transforms_augmentation)
        self.complaint_val = MarmotDataset(self.data[val_slice], transforms=self.transforms_preprocessing)
        self.complaint_test = MarmotDataset(self.data[test_slice], transforms=self.transforms_preprocessing)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.complaint_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.complaint_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.complaint_test, batch_size=self.batch_size, num_workers=self.num_workers)
