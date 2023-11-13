""" to suppress sklearn warnings. Many warning are thrown during model selection."""

import json
import os
import pathlib
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV

from experiments.comparisons.BudgetedSVMToolbox import BudgetedSVMToolbox
from experiments.comparisons.LIBIRWLS import IRWLS
from experiments.comparisons.NSSVM import NSSVM
from experiments.storage.GDriveStorage import GDriveStorage
from experiments.utils import Timer


def train_and_evaluate_NSSVM(data):
    ds = all_datasets[data["dataset"]]

    model = NSSVM(budget=data["num_sv"])
    model.fit(ds.X_train, ds.y_train)
    return {
        "dataset": data["dataset"],
        "budget": data["budget"],
        "train_time_sec": model.time_,
        "num_sv": model.num_sv_,
        "score": model.score(ds.X_test, ds.y_test),
        "params": None,
        "method": "NSSVM",
    }


def train_and_evaluate_BSGD(our_method_results):
    model = BudgetedSVMToolbox()
    ds = all_datasets[our_method_results["dataset"]]
    cv = GridSearchCV(
        model,
        param_grid={
            "budget": [our_method_results["num_sv"]],
            "gamma": [0.001, 0.01, 0.1, 1, 10],
            "kernel": [0, 2],
            "degree": [2, 5, 10],
            "strategy": [0, 1]
        },
        cv=5,
        n_jobs=1,
    )

    cv.fit(ds.X_train, ds.y_train)

    return {
        "dataset": our_method_results["dataset"],
        "budget": our_method_results["budget"],
        "train_time_sec": cv.best_estimator_.time_,
        "num_sv": cv.best_estimator_.num_sv_,
        "score": cv.best_estimator_.score(ds.X_test, ds.y_test),
        "params": cv.best_params_,
        "method": "BSGD",
    }


def train_and_evaluate_IRWLS(our_method_results):
    model = IRWLS()
    ds = all_datasets[our_method_results["dataset"]]
    cv = GridSearchCV(
        model,
        param_grid={
            "budget": [our_method_results["num_sv"]],
            "kernel": [0, 1],
            "gamma": [0.001, 0.01, 0.1, 1, 10],
            "C": [0.01, 0.1, 1, 10],
        },
        cv=5,
        n_jobs=1,
    )

    cv.fit(ds.X_train, ds.y_train)

    return {
        "dataset": our_method_results["dataset"],
        "budget": our_method_results["budget"],
        "train_time_sec": cv.best_estimator_.time_,
        "num_sv": cv.best_estimator_.num_sv_,
        "score": cv.best_estimator_.score(ds.X_test, ds.y_test),
        "params": cv.best_params_,
        "method": "IRWLS",
    }


if __name__ == "__main__":
    global all_datasets

    CWD = pathlib.Path(os.path.dirname(__file__)).absolute()
    EXPERIMENT_ID = f"{str(time.time())}_compare_5cv"

    storage = GDriveStorage()

    # retrieve last complete experiments results from our method
    our_method_results_df = pd.concat(
        [
            storage.get_result_dataframe("1688809754.2443595.json"),
            storage.get_result_dataframe("1688975220.8464673.json"),
            storage.get_result_dataframe("1689153514.3068242.json"),
        ],
        ignore_index=True,
    )
    all_datasets = {
       ds_id: storage.get_dataset_if_exists(ds_id)
       for ds_id in our_method_results_df.dataset.unique()
    }

    our_method_results_df = our_method_results_df[
        ["dataset", "budget", "train_time_min", "num_sv", "score", "budget_percentage", "params"]
    ].copy()
    our_method_results_df.loc[:, "train_time_min"] *= 60
    our_method_results_df = our_method_results_df.rename(
        columns={"train_time_min": "train_time_sec"}
    )
    our_method_results_df["method"] = "our"

    try:
        with Timer() as main_timer:
            print("BSGD")
            bsgd_res = our_method_results_df.apply(train_and_evaluate_BSGD, axis=1, result_type="expand")
            print("NSSVM")
            nssvm_res = our_method_results_df.apply(train_and_evaluate_NSSVM, axis=1, result_type="expand")
            print("IRWLS")
            irwls_res = our_method_results_df.apply(train_and_evaluate_IRWLS, axis=1, result_type="expand")

            merged = pd.concat([our_method_results_df, nssvm_res, bsgd_res, irwls_res])
            merged = merged.reset_index()

            storage.save_results(json.loads(merged.to_json()), str(EXPERIMENT_ID))
    finally:
        pass
    print("Done")

