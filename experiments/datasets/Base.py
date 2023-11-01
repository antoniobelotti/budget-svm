from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from experiments.storage.Storage import Storage

from budgetsvm.kernel import *
import numpy as np


@dataclass
class Dataset:
    id: str
    X: npt.ArrayLike
    y: npt.ArrayLike
    X_train: npt.ArrayLike
    X_test: npt.ArrayLike
    y_train: npt.ArrayLike
    y_test: npt.ArrayLike
    params: dict
    complexity_report: dict

    @classmethod
    def from_json(cls, ds_json) -> "Dataset":
        return Dataset(
            id=ds_json["id"],
            X=np.array(ds_json["X"]),
            y=np.array(ds_json["y"]),
            X_train=np.array(ds_json["X_train"]),
            X_test=np.array(ds_json["X_test"]),
            y_train=np.array(ds_json["y_train"]),
            y_test=np.array(ds_json["y_test"]),
            params=ds_json["params"],
            complexity_report=ds_json["complexity_report"],
        )

    def to_PrecomputedKernelDataset(self, kernel: Kernel):
        return PrecomputedKernelDataset(
            dataset_id=self.id,
            X_train=compute_gram_matrix(self.X_train, self.X_train, kernel),
            X_test=compute_gram_matrix(self.X_train, self.X_test, kernel),
            y_train=self.y_train,
            y_test=self.y_test,
            kernel=kernel,
        )


def compute_gram_matrix(X_train: npt.NDArray[float], X_test: npt.NDArray[float], kernel: Kernel) -> npt.NDArray[float]:
    return np.array([[kernel.compute(x1, x2) for x2 in X_train] for x1 in X_test])


@dataclass
class PrecomputedKernelDataset:
    dataset_id: str
    kernel: Kernel
    X_train: npt.NDArray[float]
    X_test: npt.NDArray[float]
    y_train: npt.NDArray[int]
    y_test: npt.NDArray[int]

    @property
    def id(self):
        return self.dataset_id + str(self.kernel)

    @classmethod
    def from_json(cls, ds_json) -> "PrecomputedKernelDataset":
        return PrecomputedKernelDataset(
            dataset_id=ds_json["dataset_id"],
            X_train=np.array(ds_json["X_train"]),
            X_test=np.array(ds_json["X_test"]),
            y_train=np.array(ds_json["y_train"]),
            y_test=np.array(ds_json["y_test"]),
            kernel=eval(ds_json["kernel"]),  # FIXME: !!!!
        )


class PrecomputedKernelDatasetBuilder:
    def __init__(self, storage: Storage, dataset: Dataset):
        self.storage = storage
        self.dataset = dataset

    def build_for(self, kernel: Kernel):
        unique_id = self.dataset.id + str(kernel)
        ds = self.storage.get_precomputed_kernel_dataset_if_exists(unique_id)
        if ds:
            return ds

        ds = self.dataset.to_PrecomputedKernelDataset(kernel)
        self.storage.save_dataset(ds)
        return ds
