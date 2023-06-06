""" to suppress sklearn warnings. Many warning are thrown during model selection."""
import sys
import traceback

from experiments.synthetic_datasets.pacman_ds import get_pacman_dataset
from experiments.synthetic_datasets.sinusoid_ds import get_sinusoid_dataset
from experiments.synthetic_datasets.sklearn_ds import get_skl_classification_dataset


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
from experiments.utils import Timer, model_selection, CustomJSONEncoder


def task_experiment_on_dataset(dataset, cfg):
    best_sv_number = None
    prev_budget = None

    results = []
    for perc in sorted(cfg.get("budget_percentages", [1.0]), reverse=True):
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
            best_model, params, score = model_selection(
                model,
                dataset.X_train,
                dataset.X_test,
                dataset.y_train,
                dataset.y_test,
                cfg["model_selection"],
            )
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
                "a_eq_c": len(best_model.alpha_eq_c_) if best_model else 0,
                "a_lt_c": len(best_model.alpha_lt_c_) if best_model else 0
            }
        )

        # if current trained model is full budget save the number of sv.
        if not best_sv_number:
            best_sv_number = len(best_model.alpha_)

    return results


def get_datasets(cfg):
    n_try_different_seed = cfg.get("n_repeat_sampling", 1)
    r_values = cfg.get("r_values", [0])
    p_values = cfg.get("p_values", [1])

    if "sinusoid" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(cfg["sinusoid"], seeds, r_values, p_values):
            yield get_sinusoid_dataset(r=r, p=p, seed=seed, **base_params)

    if "pacman" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(cfg["pacman"], seeds, r_values, p_values):
            yield get_pacman_dataset(r=r, p=p, seed=seed, **base_params)

    if "skl_classification" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2 ** 32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(cfg["skl_classification"], seeds, r_values, p_values):
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
        for ds in get_datasets(config["datasets"]):
            logging.info(f"Launching experiments on dataset f{ds.id}")
            try:
                ds_res = task_experiment_on_dataset(ds, config)
            except Exception as e:
                logging.error(traceback.format_exc())
                continue

            res.extend(ds_res)

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

    logging.info(f"Done in {main_timer.time}")
