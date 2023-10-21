from pathlib import Path
from typing import Optional

from experiments.storage.Storage import Storage
from experiments.datasets.Base import Dataset
from svm import SVC


class DummyStorage(Storage):
    def save_model(self, model: SVC, model_id: str):
        pass

    def get_dataset_if_exists(self, dataset_hash: str) -> Optional[Dataset]:
        return None

    def save_dataset(self, ds: Dataset):
        pass

    def save_results(self, res: list[dict], timestamp: str):
        pass

    def save_log(self, log_file_path: Path, timestamp: str):
        pass
