from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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
    def from_json(cls, ds_json) -> 'Dataset':
        return Dataset(
            id=ds_json["id"],
            X=np.array(ds_json["X"]),
            y=np.array(ds_json["y"]),
            X_train=np.array(ds_json["X_train"]),
            X_test=np.array(ds_json["X_test"]),
            y_train=np.array(ds_json["y_train"]),
            y_test=np.array(ds_json["y_test"]),
            params=ds_json["params"],
            complexity_report=ds_json["complexity_report"]
        )
