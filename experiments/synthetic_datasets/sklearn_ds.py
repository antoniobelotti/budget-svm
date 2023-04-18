import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_blobs, make_moons, make_classification
from sklearn.model_selection import train_test_split

from experiments.synthetic_datasets.common import cache_dataset, flip_class


@cache_dataset
def dataset_make_moons(n=300, noise=0.1, test_size=0.3, random_state=42):
    X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@cache_dataset
def dataset_make_blobs(
    n=300, cluster_std=0.2, centers=2, test_size=0.3, random_state=42
):
    X, y = make_blobs(
        n_samples=n, cluster_std=cluster_std, random_state=random_state, centers=centers
    )
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@cache_dataset
def get_skl_classification_dataset(
    n=300, dim=3, sep=1.0, clusters_per_class=2, p=1.0, r=0.0, test_size=0.3, seed=42
):
    perc_pos = 1 / (1 + p)
    perc_neg = 1 - perc_pos

    X, y = make_classification(
        n_samples=n,
        n_features=dim,
        n_informative=dim,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=clusters_per_class,
        weights=(perc_pos, perc_neg),
        flip_y=0.0,
        class_sep=sep,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=seed,
    )
    y[y == 0] = -1
    rng = np.random.default_rng(seed)
    flip_class(y, r, rng)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


if __name__ == "__main__":
    get_skl_classification_dataset(n=10**4, dim=4, sep=2.0)
