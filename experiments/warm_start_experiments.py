""" to suppress sklearn warnings. Many warning are thrown during model selection."""
import copy
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import List, Any

from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from experiments.synthetic_datasets.pacman_ds import get_pacman_dataset
from experiments.synthetic_datasets.sinusoid_ds import get_sinusoid_dataset
from experiments.synthetic_datasets.sklearn_ds import get_skl_classification_dataset
from kernel import (
    PrecomputedKernel,
    LinearKernel,
    GaussianKernel,
    PolynomialKernel,
    Kernel,
)
from optimization import ReusableGurobiSolver


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import itertools as it
import logging
import json
import math
import os
import pathlib
import pickle
import textwrap
import time
import uuid

import numpy as np

from budgetsvm.svm import SVC
from experiments.utils import Timer, CustomJSONEncoder


def model_selection_precomp_kernel(
        dataset_id: str,
        precomputed_X_train: list[list] | np.array,
        precomputed_X_test: list[list] | np.array,
        y_train: list[int] | np.array,
        y_test: list[int] | np.array,
        kernel: Kernel,
        cv: int,
        c_values: list[float],
        budget_percentages: list[float],
        seed: int = 42,
) -> list[dict]:
    print(f"Model selection on dataset {dataset_id} using {kernel}, from pid={os.getpid()}")
    solver = ReusableGurobiSolver()

    best_sv_number = None
    prev_budget = None

    results = []
    for perc in sorted(budget_percentages, reverse=True):
        budget = None
        if best_sv_number:
            budget = int(best_sv_number * perc)
            if budget == prev_budget or budget < 2:
                logging.debug(
                    f"dataset {dataset.id}: skip budget {perc * 100} either because it's <2 or equal to previous "
                    f"iteration budget"
                )
            prev_budget = budget

        model = SVC(budget=budget) if budget else SVC()
        model_name = "full_budget" if not best_sv_number else f"{perc:.2f}_budget"

        logging.debug(
            f"Dataset {dataset.id[-10:]} Budget {perc * 100}% - Launching model selection"
        )

        with Timer() as t:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

            # ! n_jobs must be 1. This is already running in a dedicated process
            cvgrid = GridSearchCV(
                model,
                {"C": c_values, "kernel": [kernel]},
                refit=True,
                verbose=0,
                cv=skf,
                n_jobs=1,
            )
            best_model, params, score = None, None, 0
            try:
                cvgrid.fit(precomputed_X_train, y_train, solver=solver)
                test_score = cvgrid.score(precomputed_X_test, y_test)
                best_model, params, score = (
                    cvgrid.best_estimator_,
                    cvgrid.best_params_,
                    test_score,
                )
            except FitFailedWarning:
                pass
            except Exception as e:
                logging.error("GridSearchCV failed with unexpected error.")
                logging.error(traceback.format_exc())

        logging.debug(
            f"Dataset {dataset.id[-10:]} Budget {perc * 100}% - "
            f"Model selection took {t.time} seconds"
        )

        if not hasattr(best_model, "optimal_"):
            logging.error(
                f"Trained model {best_model} has no attribute optimal_. This should not happen."
            )
            best_model.optimal_ = False

        best_model_uuid = uuid.uuid4()
        results.append(
            {
                "dataset": dataset.id,
                "model_UUID": best_model_uuid,
                "model": best_model,
                "model_name": model_name,
                "optimal": best_model.optimal_ if best_model else None,
                "params": params,
                "score": score,
                "budget": budget if budget else math.inf,
                "num_sv": len(best_model.alpha_) if best_model else None,
                "train_time": t.time,
            }
        )

        # if current trained model is full budget save the number of sv.
        if not best_sv_number:
            best_sv_number = len(best_model.alpha_)

    return results


def process_dataset(dataset, cfg) -> list[dict]:
    """
    Foreach parameter train a model on dataset and return the best performing model
    """
    logging.info(f"Processing dataset {dataset.id}")
    executor = ProcessPoolExecutor(max_workers=os.cpu_count())
    futures = []
    logging.info(f"using an executor with {executor._max_workers} workers")

    for kernel_name, kernel_parameters in cfg["model_selection"]["kernels"].items():
        # linear kernel has no parameters
        kernel_parameters = kernel_parameters if kernel_parameters else [None]
        for v in kernel_parameters:
            match kernel_name:
                case "gaussian":
                    original_kernel = GaussianKernel(v)
                case "polynomial":
                    original_kernel = PolynomialKernel(v)
                case _:
                    original_kernel = LinearKernel()

            kernel = PrecomputedKernel(original_kernel=original_kernel)
            precomputed_X_train = np.array(
                [
                    [kernel.compute(x, y) for y in dataset.X_train]
                    for x in dataset.X_train
                ]
            )
            precomputed_X_test = np.array(
                [
                    [kernel.compute(x, y) for y in dataset.X_train]
                    for x in dataset.X_test
                ]
            )

            futures.append(executor.submit(
                model_selection_precomp_kernel,
                dataset.id,
                precomputed_X_train,
                precomputed_X_test,
                dataset.y_train,
                dataset.y_test,
                kernel,
                config["model_selection"]["cv"],
                config["model_selection"]["C"],
                config["budget_percentages"],
            ))

    # blocks until all tasks are completed
    executor.shutdown(wait=True)

    results = []
    for ft in futures:
        results.extend(ft.result())
    return results
    #return reduce(lambda r, n: r.extend(n.result), futures, [])


def get_datasets(cfg):
    n_try_different_seed = cfg.get("n_repeat_sampling", 1)
    r_values = cfg.get("r_values", [0])
    p_values = cfg.get("p_values", [1])

    if "sinusoid" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2 ** 32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
                cfg["sinusoid"], seeds, r_values, p_values
        ):
            yield get_sinusoid_dataset(r=r, p=p, seed=seed, **base_params)

    if "pacman" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2 ** 32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
                cfg["pacman"], seeds, r_values, p_values
        ):
            yield get_pacman_dataset(r=r, p=p, seed=seed, **base_params)

    if "skl_classification" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2 ** 32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
                cfg["skl_classification"], seeds, r_values, p_values
        ):
            yield get_skl_classification_dataset(r=r, p=p, seed=seed, **base_params)


if __name__ == "__main__":
    with Timer() as main_timer:
        CWD = pathlib.Path(os.path.dirname(__file__)).absolute()

        BASE_DIR_PATH = pathlib.Path(CWD / "results")
        BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)

        NOW = time.time()
        OUT_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}.json")

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(BASE_DIR_PATH / f"{NOW}.log"),
                logging.StreamHandler(),
            ],
        )

        with open(pathlib.Path(CWD / "dev_exp_config_small.json"), "rb") as f:
            config = json.load(f)

        logging.info(
            textwrap.dedent(
                f"""
                Running experiment with the following configuration:

                {json.dumps(config, indent=4)}
                """
            )
        )

        res = []
        for dataset in get_datasets(config["datasets"]):
            res.extend(process_dataset(dataset, config))

        logging.info("Saving models on disk")
        pathlib.Path(BASE_DIR_PATH / "models").mkdir(exist_ok=True)
        for r in res:
            model_name = r.get("model_UUID", "?")
            model = r.pop("model", None)
            if model is None:
                continue
            with open(BASE_DIR_PATH / "models" / f"{model_name}.pkl", "wb") as f:
                pickle.dump(model, f)

        logging.info("Saving results on disk")
        with open(OUT_FILE_PATH, "w+") as f:
            f.write(json.dumps(res, cls=CustomJSONEncoder))

        logging.info(res)

    logging.info(f"Done in {main_timer.time}")
