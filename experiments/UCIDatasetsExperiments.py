from collections.abc import Iterator

import numpy as np
import requests
from sklearn.datasets import load_svmlight_file

from budgetsvm.svm import SVC
from experiments.BaseExperiment import BaseExperiment
from experiments.datasets.Base import Dataset
from experiments.storage.GDriveStorage import GDriveStorage


class SmallUCIDatasetsExperiments(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
    EXPERIMENT_FILE_SUFFIX = "_small_uci_datasets_our_model"

    def get_cvgrid_parameter_dict(self) -> dict:
        return {}

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


if __name__ == "__main__":
    exp = SmallUCIDatasetsExperiments(GDriveStorage())
    exp.run()
