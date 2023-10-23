import hashlib
import inspect
from abc import abstractmethod, ABC
from typing import Callable

import numpy as np
import numpy.typing as npt
import problexity as px
from sklearn.model_selection import train_test_split

from experiments.storage.Storage import Storage
from experiments.datasets.Base import Dataset


class BaseDatasetBuilder(ABC):
    rng: np.random.Generator
    storage: Storage

    def __init__(self, storage: Storage):
        self.storage = storage

    @abstractmethod
    def get_population(self, population_size: int, **kwargs) -> npt.NDArray[float]:
        raise NotImplementedError

    @abstractmethod
    def get_labeling_fn(self, **kwargs) -> Callable[[npt.NDArray[float]], npt.NDArray[float]]:
        raise NotImplementedError

    def build(
        self,
        n: int = 1000,
        r: float = 0,
        p: float = 1,
        seed: int = 42,
        test_size: float = 0.3,
        dim: int = 2,
        **kwargs,
    ) -> Dataset:
        bound = inspect.signature(self.build).bind(n, r, p, seed, test_size, dim, **kwargs)
        bound.apply_defaults()
        dataset_generation_parameters = bound.arguments
        binded_kwargs = dataset_generation_parameters.pop("kwargs")
        dataset_generation_parameters = {**bound.arguments, **binded_kwargs}

        params_str = str(sorted(dataset_generation_parameters.items())).encode("utf-8")

        dataset_hash = hashlib.md5(params_str).hexdigest()

        ds = self.storage.get_dataset_if_exists(dataset_hash)
        if ds:
            return ds

        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # create a population 10 times bigger than desired dataset size
        X = self.get_population(n * 10, **kwargs)
        y = self.get_labeling_fn(**kwargs)(*X.T)

        # sample with desired class balance
        mask = self.get_indices_with_class_balance(y, n, p)
        X = X[mask]
        y = y[mask]

        # introduce desired amount of noise
        self.flip_class(y, r)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        cc = px.ComplexityCalculator()
        y_copy = y.copy()
        y_copy[y_copy == -1] = 0  # problexity wants 0,1 labels...
        cc.fit(X, y_copy)
        complexity_report = cc.report()
        complexity_report.pop("classes", None)
        complexity_report.pop("n_classes", None)

        ds = Dataset(
            id=dataset_hash,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=dataset_generation_parameters,
            complexity_report=complexity_report,
        )

        self.storage.save_dataset(ds)

        return ds

    def flip_class(self, y: npt.ArrayLike, r: float):
        """
        Randomly select a subset of samples labels and flip their value. Labels are assumed to be {-1,1}.

        :param y: iterable collection of label values
        :param r: percentage of positive samples to be randomly selected and class-swapped
                  with an equal number of negative samples
        :param rng: numpy Generator class for anything random
        """

        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == -1)[0]

        flip_size = int(r * len(positive_indices))

        pos_flip_idx = self.rng.choice(positive_indices, size=flip_size, replace=False)
        neg_flip_idx = self.rng.choice(negative_indices, size=flip_size, replace=False)

        y[pos_flip_idx] = -y[pos_flip_idx]
        y[neg_flip_idx] = -y[neg_flip_idx]

    def get_indices_with_class_balance(self, y: npt.ArrayLike, n: int, p: float):
        """
        Returns a list of indices that represent a sample with the desired class imbalance.

        :param y: iterable collection of label values
        :param n: size of sample to produce
        :param p: ratio between the number of positive samples and negative samples we want to obtain
        :param rng: numpy Generator class for anything random

        :return: numpy array of int indices
        """
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == -1)[0]

        num_pos = int(n / (p + 1))
        num_neg = n - num_pos

        return np.concatenate(
            [
                self.rng.choice(positive_indices, size=num_pos, replace=False),
                self.rng.choice(negative_indices, size=num_neg, replace=False),
            ]
        )
