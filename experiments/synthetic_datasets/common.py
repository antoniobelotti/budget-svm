import hashlib
import inspect
import json
import os
import pathlib

from types import SimpleNamespace

import numpy as np
import problexity as px

from experiments.utils import CustomJSONEncoder


def load_dataset(path):
    """Also used in jupyter notebooks"""
    with open(path, "rb") as f:
        ds = json.load(f)

        # manually convert lists to numpy arrays. There are lists nested deeper but there's no need to convert them
        ds = {k: np.array(v) if type(v) == list else v for k, v in ds.items()}

        return SimpleNamespace(**ds)


def __get_dataset_hash(params):
    params_str = str(sorted(params.items())).encode("utf-8")
    return hashlib.md5(params_str).hexdigest()


def cache_dataset(dataset_create):
    """
    Decorator function that loads a dataset if it already exists, else creates it and saves it.
    Decorated function must return X_train, X_test, y_train, y_test
    """

    def wrapper(*args, **kwargs):
        DIR_PATH = pathlib.Path(os.path.dirname(__file__)).absolute() / pathlib.Path(
            "datasets"
        )
        DIR_PATH.mkdir(parents=True, exist_ok=True)

        # default values do not appear in *args **kwargs but are needed to create hash
        #  1) extract defaults using introspection module
        #  2) merge defaults with *args **kwargs to obtain full list of parameters
        bound = inspect.signature(dataset_create).bind(*args, **kwargs)
        bound.apply_defaults()
        all_params = bound.arguments

        ds_id = __get_dataset_hash(all_params)
        path = DIR_PATH / f"{ds_id}.json"
        if pathlib.Path(path).exists():
            return load_dataset(path)

        X_train, X_test, y_train, y_test = dataset_create(*args, **kwargs)
        X = np.append(X_train, X_test, axis=0)
        y = np.append(y_train, y_test, axis=0)

        # Data complexity measures
        cc = px.ComplexityCalculator()
        y_copy = y.copy()
        y_copy[y_copy == -1] = 0  # problexity wants 0,1 labels...
        cc.fit(X, y_copy)
        complexity_report = cc.report()
        complexity_report.pop("classes", None)
        complexity_report.pop("n_classes", None)

        ds = {
            "id": ds_id,
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "params": all_params,
            "complexity_report": complexity_report,
        }

        with open(path, "w") as f:
            json.dump(ds, f, cls=CustomJSONEncoder)

        ds = SimpleNamespace(**ds)  # to use dict with dot notation

        return ds

    return wrapper


def flip_class(y: np.array, r: float, rng: np.random.Generator):
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

    pos_flip_idx = rng.choice(positive_indices, size=flip_size, replace=False)
    neg_flip_idx = rng.choice(negative_indices, size=flip_size, replace=False)

    y[pos_flip_idx] = -y[pos_flip_idx]
    y[neg_flip_idx] = -y[neg_flip_idx]


def get_indices_with_class_balance(
    y: np.array, n: int, p: float, rng: np.random.Generator
):
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
            rng.choice(positive_indices, size=num_pos, replace=False),
            rng.choice(negative_indices, size=num_neg, replace=False),
        ]
    )
