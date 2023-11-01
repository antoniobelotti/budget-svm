from __future__ import print_function

import gzip
import json
import os
import pathlib
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from experiments.storage.Storage import Storage
from experiments.datasets.Base import Dataset, PrecomputedKernelDataset
from experiments.utils import CustomJSONEncoder, Timer
from budgetsvm.svm import SVC


class LocalStorage(Storage):
    def __init__(self, base=pathlib.Path("/tmp/ahah")):
        self.base = base

    def save_model(self, model: SVC, model_id: str):
        with open(self.base / f"{model_id}.pkl", "wb") as f:
            pickle.dump(model, f)

    def save_dataset(self, ds: Dataset | PrecomputedKernelDataset):
        content = bytes(json.dumps(ds, cls=CustomJSONEncoder), "utf-8")
        with gzip.open(self.base / f"{ds.id}.json.gz", "wb") as f:
            f.write(content)

    def save_results(self, res: list[dict], timestamp: str):
        tmp_filepath = self.base / f"{timestamp}.json"
        with open(tmp_filepath, "w") as f:
            f.write(json.dumps(res, cls=CustomJSONEncoder))

    def save_log(self, log_file_path: Path, timestamp: str):
        os.rename(log_file_path, self.base / f"{timestamp}.log")

    def get_dataset_if_exists(self, ds_id: str) -> Optional[Dataset]:
        ds_file = self.base / f"{ds_id}.json.gz"
        if not ds_file.exists():
            return None
        with gzip.open(ds_file, "rb") as f:
            content = f.read()
        ds = json.loads(content)
        return Dataset.from_json(ds)

    def get_precomputed_kernel_dataset_if_exists(self, ds_id: str) -> Optional[PrecomputedKernelDataset]:
        ds_file = self.base / f"{ds_id}.json.gz"
        if not ds_file.exists():
            return None
        with gzip.open(ds_file, "rb") as f:
            content = f.read()
        ds = json.loads(content)

        return PrecomputedKernelDataset.from_json(ds)

    def get_log_file(self, log_file_id: str) -> str:
        ds_file = self.base / f"{log_file_id}.log"
        if not ds_file.exists():
            raise FileNotFoundError()
        with open(ds_file, "r") as f:
            return f.read()

    def get_results_json(self, experiment_id: str) -> list[dict]:
        res_file = self.base / f"{experiment_id}.json"
        if not res_file.exists():
            raise FileNotFoundError()
        with open(res_file, "r") as f:
            return json.load(f)

    def get_model(self, model_uuid) -> BaseEstimator:
        model_file = self.base / f"{model_uuid}.pkl"
        if not model_file.exists():
            raise FileNotFoundError()
        with open(model_file, "rb") as f:
            return pickle.load(f)

    def get_result_dataframe(self, experiment_id: str) -> pd.DataFrame:
        pass

