from abc import abstractmethod
from pathlib import Path
from typing import Protocol, Optional

from experiments.synthetic_datasets.Dataset import Dataset
from budgetsvm.svm import SVC


class Storage(Protocol):
    """This class defines the interface for every action related with persistence of flat files in the context of
    this project experiments. Save and retrieve models, datasets, results, logs."""

    @abstractmethod
    def save_model(self, model: SVC, model_id: str):
        """Save an SVC model

        Args:
            model_id (str): the model id
            model (SVC): the model to save
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_if_exists(self, dataset_hash: str) -> Optional[Dataset]:
        """Download the dataset identified by dataset_hash if exists, else return None

        Args:
            dataset_hash (str): hash string identifying the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def save_dataset(self, ds: Dataset):
        """Save a Dataset object

        Args:
            ds (Dataset): dataset to save
        """
        raise NotImplementedError

    @abstractmethod
    def save_results(self, res: list[dict], timestamp: str):
        """Save the results of an experiment

        Args:
            res (list[dict]): results data
            timestamp (str): Unix timestamp when the experiment was launched
        """
        raise NotImplementedError

    @abstractmethod
    def save_log(self, log_file_path: Path, timestamp:str):
        """Save the logs of an experiment

        Args:
            log_file_path (Path): Path for the log file
            timestamp (str): Unix timestamp when the experiment was launched
        """
        raise NotImplementedError
