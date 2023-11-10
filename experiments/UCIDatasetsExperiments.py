import bz2
from collections.abc import Iterator

import numpy as np
import requests
from sklearn.datasets import load_svmlight_file

from budgetsvm.kernel import PrecomputedKernel, GaussianKernel, PolynomialKernel
from budgetsvm.svm import SVC
from experiments.BaseExperiment import BaseExperiment
from experiments.datasets.Base import Dataset
from experiments.storage.GDriveStorage import GDriveStorage


class OurMethodSmallUCIDatasetsExperiments(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    EXPERIMENT_FILE_SUFFIX = "_small_uci_datasets_our_model"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        datasets = [("a1a", 123), ("a2a", 123), ("a3a", 123)]
        for name, n_features in datasets:
            cached_ds = self.storage.get_dataset_if_exists(name)
            if cached_ds:
                yield cached_ds
                continue

            train_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}").text
            test_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}.t").text
            with open(f"/tmp/{name}", "w") as f:
                f.write(train_content)
            with open(f"/tmp/{name}.t", "w") as f:
                f.write(test_content)

            X_train, y_train = load_svmlight_file(f"/tmp/{name}", zero_based=False, n_features=n_features)
            X_test, y_test = load_svmlight_file(f"/tmp/{name}.t", zero_based=False, n_features=n_features)
            X_train = X_train.toarray()
            X_test = X_test.toarray()

            X = np.concatenate([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            ds = Dataset(
                id=name,
                X=X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                params=None,
                complexity_report=None,
            )
            self.storage.save_dataset(ds)
            yield ds


class OurMethodGisetteDatasetExperiments(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    EXPERIMENT_FILE_SUFFIX = "_gisette_dataset_our_model"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        train_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2").content
        test_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2").content
        train_content = bz2.decompress(train_content)
        test_content = bz2.decompress(test_content)

        with open("/tmp/gisette", "wb") as f:
            f.write(train_content)
        with open("/tmp/gisette.t", "wb") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file("/tmp/gisette", zero_based=False, n_features=5000)
        X_test, y_test = load_svmlight_file("/tmp/gisette.t", zero_based=False, n_features=5000)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id="gisette",
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


if __name__ == "__main__":
    #exp = OurMethodSmallUCIDatasetsExperiments(GDriveStorage())
    exp = OurMethodSmallUCIDatasetsExperiments(GDriveStorage())
    exp.run()
