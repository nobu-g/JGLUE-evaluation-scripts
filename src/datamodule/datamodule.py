from dataclasses import fields, is_dataclass
from typing import Any, Optional, Union

import hydra
import lightning
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == TrainerFn.FITTING:
            self.train_dataset = hydra.utils.instantiate(self.cfg.train)
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TESTING):
            self.valid_dataset = hydra.utils.instantiate(self.cfg.valid)
        if stage == TrainerFn.TESTING:
            self.test_dataset = hydra.utils.instantiate(self.cfg.test)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._get_dataloader(dataset=self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.valid_dataset is not None
        return self._get_dataloader(self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataclass_data_collator,
            pin_memory=True,
        )


def dataclass_data_collator(features: list[Any]) -> dict[str, Union[Tensor, list[str]]]:
    first: Any = features[0]
    assert is_dataclass(first), "Data must be a dataclass"
    batch: dict[str, Union[Tensor, list[str]]] = {}
    for field in fields(first):
        feats = [getattr(f, field.name) for f in features]
        batch[field.name] = torch.as_tensor(feats)
    return batch
