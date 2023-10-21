from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kernel import Kernel, GaussianKernel


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
            # FIXME: there's some repeated computation but...
            X=compute_gram_matrix(self.X, kernel),
            X_train=compute_gram_matrix(self.X_train, kernel),
            X_test=compute_gram_matrix(self.X_test, kernel),
            y=self.y,
            y_train=self.y_train,
            y_test=self.y_test,
            kernel=kernel,
        )


def compute_gram_matrix(X: npt.NDArray[float], kernel: Kernel) -> npt.NDArray[float]:
    # TODO: unit test !!
    return np.array([[kernel.compute(x1, x2) for x2 in X] for x1 in X])


@dataclass
class PrecomputedKernelDataset:
    dataset_id: str
    kernel: Kernel
    X: npt.NDArray[float]
    y: npt.NDArray[int]
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
            X=np.array(ds_json["X"]),
            y=np.array(ds_json["y"]),
            X_train=np.array(ds_json["X_train"]),
            X_test=np.array(ds_json["X_test"]),
            y_train=np.array(ds_json["y_train"]),
            y_test=np.array(ds_json["y_test"]),
            kernel=eval(ds_json["kernel"]),  # FIXME: !!!!
        )
