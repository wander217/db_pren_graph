from torch.utils.data import DataLoader, dataloader
from .det_dataset import DBDataset
from typing import Dict


class DBLoader:
    def __init__(self,
                 num_workers: int, batch_size: int,
                 drop_last: bool, shuffle: bool,
                 pin_memory: bool, dataset: Dict):
        self.dataset: DBDataset = DBDataset(**dataset)
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=dataloader.default_collate
        )
