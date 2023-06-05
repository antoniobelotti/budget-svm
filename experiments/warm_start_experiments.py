""" to suppress sklearn warnings. Many warning are thrown during model selection."""
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue

from numpy._typing import ArrayLike
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
    precomputed_X_train: ArrayLike,
    precomputed_X_test: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
    kernel: Kernel,
    cv: int,
    c_values: list[float],
    budget_percentages: list[float],
    queue: Queue,
    seed: int = 42
):
    print(f"model selection on dataset {dataset_id} with kernel {kernel}")
    # solver = ReusableGurobiSolver()

    best_sv_number = None
    prev_budget = None

    for perc in sorted(budget_percentages, reverse=True):
        budget = None
        if best_sv_number:
            budget = int(best_sv_number * perc)
            if budget == prev_budget or budget < 2:
                print(
                    f"dataset {dataset.id}: skip budget {perc * 100} either because it's <2 or equal to previous "
                    f"iteration budget"
                )
            prev_budget = budget

        model = SVC(budget=budget) if budget else SVC()
        model_name = "full_budget" if not best_sv_number else f"{perc:.2f}_budget"

        print(
            f"Dataset {dataset_id[-10:]} Budget {perc * 100}% - Launching model selection"
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
                cvgrid.fit(precomputed_X_train, y_train)#, solver=solver)
                test_score = cvgrid.score(precomputed_X_test, y_test)
                best_model, params, score = (
                    cvgrid.best_estimator_,
                    cvgrid.best_params_,
                    test_score,
                )
            except FitFailedWarning:
                pass
            except Exception as e:
                print("GridSearchCV failed with unexpected error.")
                print(traceback.format_exc())

        if best_model is None:
            if best_sv_number is None:
                return # model selection failed on regular unconstrained model

            # a budget constrained model selection failed. Continue and try another budget value
            continue

        print(
            f"Dataset {dataset_id[-10:]} Budget {perc * 100}% - "
            f"Model selection took {t.time} seconds"
        )

        if not hasattr(best_model, "optimal_"):
            print(
                f"Trained model {best_model} has no attribute optimal_. This should not happen."
            )
            best_model.optimal_ = False

        best_model_uuid = uuid.uuid4()
        queue.put([
            {
                "dataset": dataset_id,
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
        ])

        # if current trained model is full budget save the number of sv.
        if not best_sv_number:
            best_sv_number = len(best_model.alpha_)

    # del solver.env


def process_dataset(dataset, cfg) -> list[dict]:
    """
    Foreach parameter train a model on dataset and return the best performing model
    """
    print(f"Processing dataset {dataset.id}")

    processes = []
    results_queue = Queue()

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

            processes.append(
                Process(
                    target=model_selection_precomp_kernel,
                    args=(
                        dataset.id,
                        precomputed_X_train,
                        precomputed_X_test,
                        dataset.y_train,
                        dataset.y_test,
                        kernel,
                        config["model_selection"]["cv"],
                        config["model_selection"]["C"],
                        config["budget_percentages"],
                        results_queue
                    ),
                )
            )

    max_concurrent_processes = os.cpu_count()
    # for i in range(0,len(processes), max_concurrent_processes):
    #     for p in processes[i: i+max_concurrent_processes]:
    #         p.start()
    #     for p in processes[i: i + max_concurrent_processes]:
    #         p.join()

    running = []
    while len(processes) or len(running):
        for rp in running:
            rp.join(0)
            if not rp.is_alive():
                # process done
                running.remove(rp)
        if len(running) < max_concurrent_processes and len(processes):
            new_p = processes.pop()
            new_p.start()
            running.append(new_p)

    results = results_queue.get()
    return results


def get_datasets(cfg):
    n_try_different_seed = cfg.get("n_repeat_sampling", 1)
    r_values = cfg.get("r_values", [0])
    p_values = cfg.get("p_values", [1])

    if "sinusoid" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
            cfg["sinusoid"], seeds, r_values, p_values
        ):
            yield get_sinusoid_dataset(r=r, p=p, seed=seed, **base_params)

    if "pacman" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
            cfg["pacman"], seeds, r_values, p_values
        ):
            yield get_pacman_dataset(r=r, p=p, seed=seed, **base_params)

    if "skl_classification" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
            cfg["skl_classification"], seeds, r_values, p_values
        ):
            yield get_skl_classification_dataset(r=r, p=p, seed=seed, **base_params)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    with Timer() as main_timer:
        CWD = pathlib.Path(os.path.dirname(__file__)).absolute()

        BASE_DIR_PATH = pathlib.Path(CWD / "results")
        BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)

        NOW = time.time()
        OUT_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}.json")

        with open(pathlib.Path(CWD / "dev_exp_config_small.json"), "rb") as f:
            config = json.load(f)

        print(
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

        print(res)
        print("Saving models on disk")
        pathlib.Path(BASE_DIR_PATH / "models").mkdir(exist_ok=True)
        for r in res:
            model_name = r.get("model_UUID", "?")
            model = r.pop("model", None)
            if model is None:
                continue
            with open(BASE_DIR_PATH / "models" / f"{model_name}.pkl", "wb") as f:
                pickle.dump(model, f)

        print("Saving results on disk")
        print(res)
        with open(OUT_FILE_PATH, "w+") as f:
            f.write(json.dumps(res, cls=CustomJSONEncoder))

    print(f"Done in {main_timer.time}")

experiments / results / 1685449053.3788764.json
experiments / results / 1685455690.3525748.json
experiments / results / 1685626882.9574878.json