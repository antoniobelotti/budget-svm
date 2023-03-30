import hashlib
import inspect
import json
import os
import pathlib

from types import SimpleNamespace

import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split


def load_dataset(path):
    """Also used in jupyter notebooks """
    with open(path, "rb") as f:
        ds = json.load(f)
        ds = {k: np.array(v) if type(v) == list else v for k, v in ds.items()}
        return SimpleNamespace(**ds)


def get_dataset_hash(params):
    params_str = str(sorted(params.items())).encode('utf-8')
    return hashlib.md5(params_str).hexdigest()


def cache(dataset_create):
    """
    Decorator function that loads a dataset if it already exists, else creates it and saves it.
    Decorated function must return X_train, X_test, y_train, y_test
    """

    def wrapper(*args, **kwargs):
        DIR_PATH = pathlib.Path(os.path.dirname(__file__)).absolute() / pathlib.Path('datasets')
        DIR_PATH.mkdir(parents=True, exist_ok=True)

        # default values do not appear in *args **kwargs but are needed to create hash
        #  1) extract defaults using introspection module
        #  2) merge defaults with *args **kwargs to obtain full list of parameters
        bound = inspect.signature(dataset_create).bind(*args, **kwargs)
        bound.apply_defaults()
        all_params = bound.arguments

        ds_id = get_dataset_hash(all_params)
        path = DIR_PATH / f"{ds_id}.json"
        if pathlib.Path(path).exists():
            return load_dataset(path)

        X_train, X_test, y_train, y_test = dataset_create(*args, **kwargs)

        ds = {
            "id": ds_id,
            "X": np.append(X_train, X_test, axis=0),
            "y": np.append(y_train, y_test, axis=0),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "params": all_params
        }

        with open(path, "w") as f:
            serializable_ds = {k: v.tolist() if type(v) == np.ndarray else v for k, v in ds.items()}
            json.dump(serializable_ds, f)

        ds = SimpleNamespace(**ds)  # to use dict with dot notation

        return ds

    return wrapper


def get_labeling_function(beta, rho, theta):
    df = lambda x: 1 / (1 + np.exp(-beta * (x - 0.5))) + rho * np.sin(2 * np.pi * theta * x)

    def lf(x0, x1):
        return np.sign(df(x0) - x1)

    return lf


@cache
def dataset_from_custom_function(beta=1, rho=.2, theta=10, n=300, test_size=.3, random_state=42):
    labeling_function = get_labeling_function(beta, rho, theta)

    np.random.seed(random_state)
    X = np.random.uniform(size=(n, 2), low=0, high=1)
    y = labeling_function(*X.T)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@cache
def unbalanced_dataset_from_custom_function(beta=1, rho=.2, theta=10, population_size=1000, sample_size=300,
                                            perc_positives=.5, test_size=.3, random_state=42):
    labeling_function = get_labeling_function(beta, rho, theta)

    np.random.seed(random_state)
    X = np.random.uniform(size=(population_size, 2), low=0, high=1)
    y = labeling_function(*X.T)

    positive_samples_size = int(sample_size * perc_positives)
    negative_samples_size = int(sample_size * (1 - perc_positives))

    positive_population_idxs = np.where(y == 1)[0]
    negative_population_idxs = np.where(y == -1)[0]

    positive_sample_indices = np.random.choice(positive_population_idxs, positive_samples_size, replace=False)
    negative_sample_indices = np.random.choice(negative_population_idxs, negative_samples_size, replace=False)

    sample_X = np.append(X[positive_sample_indices], X[negative_sample_indices], axis=0)
    sample_y = np.append(y[positive_sample_indices], y[negative_sample_indices], axis=0)

    return train_test_split(sample_X, sample_y, test_size=test_size, random_state=random_state)


@cache
def dataset_make_moons(n=300, noise=.1, test_size=.3, random_state=42):
    X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@cache
def dataset_make_blobs(n=300, cluster_std=.2, centers=2, test_size=.3, random_state=42):
    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, random_state=random_state, centers=centers)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_unbalanced_datasets():
    custom_configurations = [
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.6},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.7},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.8},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.9},

        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.6},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.7},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.8},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.9},

        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.6},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.7},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.8},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.9},

        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.6},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.7},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.8},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.9},
    ]
    for cfg in custom_configurations:
        yield unbalanced_dataset_from_custom_function(**cfg)


def generate_datasets():
    CUSTOM_CONFIGURATIONS = [
        {"beta": 1, "rho": 0.2, "theta": 10},
        {"beta": 95, "rho": 0.2, "theta": 10},
        {"beta": 0, "rho": 0.3, "theta": 3},
        {"beta": 0, "rho": 0.1, "theta": 8}
    ]
    for cfg in CUSTOM_CONFIGURATIONS:
        yield dataset_from_custom_function(**cfg)

    """
    # Generate dataset using scikit learn
    MOONS_CONFIGURATIONS = [
        {"noise": .1},
        {"noise": .2},
    ]
    for cfg in MOONS_CONFIGURATIONS:
        yield dataset_make_moons(**cfg)

    BLOBS_CONFIGURATIONS = [
        {"cluster_std": .6},
        {"cluster_std": 2},
    ]
    for cfg in BLOBS_CONFIGURATIONS:
        yield dataset_make_blobs(**cfg)
    """


#if __name__ == "__main__":
    # df = lambda x: 1 / (1 + np.exp(-95 * (x - 0.5))) + .2 * np.sin(2 * np.pi * 10 * x)

    # x_df = np.linspace(0, 1, 500)
    # y_df = df(x_df)

    # plt.plot(x_df, y_df)

    # ds = dataset_from_custom_function(95, .2, 10)
    # plt.plot(x_df, y_df)
    # plt.scatter(*ds.X.T, c=ds.y)
    # plt.show()

    # ds = dataset_make_moons(noise=.2)
    # plt.scatter(*ds.X.T, c=ds.y)
    # plt.show()

    # ds = dataset_make_blobs(cluster_std=3)
    # plt.scatter(*ds.X.T, c=ds.y)
    # plt.show()

    # ds = unbalanced_dataset_from_custom_function(95, .2, 10, perc_positives=.9)
    # plt.scatter(*ds.X.T, c=ds.y)
    # plt.show()

    # ds = dataset_make_blobs(n=50, cluster_std=2, centers=2, random_state=42)
    # plt.scatter(*ds.X.T, c=ds.y)
    # plt.show()


""""
    custom_configurations = [
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.6},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.7},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.8},
        {"beta": 1, "rho": 0.2, "theta": 10, "perc_positives": 0.9},

        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.6},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.7},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.8},
        {"beta": 95, "rho": 0.2, "theta": 10, "perc_positives": 0.9},

        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.6},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.7},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.8},
        {"beta": 0, "rho": 0.3, "theta": 3, "perc_positives": 0.9},

        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.6},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.7},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.8},
        {"beta": 0, "rho": 0.1, "theta": 8, "perc_positives": 0.9},
    ]
    for cfg in custom_configurations:
        unbalanced_dataset_from_custom_function(**cfg)
"""
