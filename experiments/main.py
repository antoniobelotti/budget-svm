""" to suppress sklearn warnings. Many warning are thrown during model selection."""
import traceback

from experiments.storage.GDriveStorage import GDriveStorage
from experiments.storage.Storage import Storage
from experiments.synthetic_datasets.PacmanDatasetBuilder import PacmanDatasetBuilder
from experiments.synthetic_datasets.SinusoidDatasetBuilder import SinusoidDatasetBuilder


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
import textwrap
import time
import uuid

import numpy as np

from budgetsvm.svm import SVC
from experiments.utils import Timer, model_selection


def task_experiment_on_dataset(dataset, cfg):
    logger = logging.getLogger("experiments")

    best_sv_number = None
    prev_budget = None

    results = []
    for perc in sorted(cfg.get("budget_percentages", [1.0]), reverse=True):
        budget = None
        if best_sv_number:
            budget = int(best_sv_number * perc)
            if budget == prev_budget or budget < 2:
                logger.debug(
                    f"dataset {dataset.id}: skip budget {perc * 100} either because it's <2 or equal to previous "
                    f"iteration budget"
                )
            prev_budget = budget

        model = SVC(budget=budget) if budget else SVC()
        model_name = "full_budget" if not best_sv_number else f"{perc:.2f}_budget"

        logger.debug(
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
        logger.debug(
            f"Dataset {dataset.id[-10:]} Budget {perc * 100}% - "
            f"Model selection took {t.time} seconds"
        )

        if best_model is None:
            logging.warning("model selection return no model")

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
                "s_pos": best_model.s_pos_ if best_model else 0,
                "s_neg": best_model.s_neg_ if best_model else 0,
                "b_pos": best_model.b_pos_ if best_model else 0,
                "b_neg": best_model.b_neg_ if best_model else 0,
                "mip_gap": best_model.mip_gap_ if best_model else math.inf
            }
        )

        # if current trained model is full budget save the number of sv.
        if not best_sv_number and best_model is not None:
            best_sv_number = len(best_model.alpha_)

    return results


def get_datasets(cfg: dict, s: Storage):
    n_try_different_seed = cfg.get("n_repeat_sampling", 1)
    r_values = cfg.get("r_values", [0])
    p_values = cfg.get("p_values", [1])

    if "sinusoid" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
            cfg["sinusoid"], seeds, r_values, p_values
        ):
            if "gamma" not in base_params:
                base_params["gamma"] = 10
            yield SinusoidDatasetBuilder(s).build(r=r, p=p, seed=seed, **base_params)

    if "pacman" in cfg:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)
        for base_params, seed, r, p in it.product(
            cfg["pacman"], seeds, r_values, p_values
        ):
            yield PacmanDatasetBuilder(s).build(r=r, p=p, seed=seed, **base_params)


if __name__ == "__main__":
    CWD = pathlib.Path(os.path.dirname(__file__)).absolute()
    EXPERIMENT_ID = str(time.time())
    TMP_LOG_FILE_PATH = CWD / f"{EXPERIMENT_ID}.log"

    with open(pathlib.Path(CWD / "dev_exp_config_small.json"), "rb") as f:
        config = json.load(f)

    logger = logging.getLogger("experiments")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s[%(levelname)s] \t%(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    fh = logging.FileHandler(TMP_LOG_FILE_PATH)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logger.info(
        textwrap.dedent(
            f"""
            Running experiment with the following configuration:

            {json.dumps(config, indent=4)}
            """
        )
    )

    try:
        with Timer() as main_timer:
            storage = GDriveStorage()

            res = []
            for ds in get_datasets(config["datasets"], storage):
                logger.info(f"Launching experiments on dataset f{ds.id}")
                try:
                    ds_res = task_experiment_on_dataset(ds, config)
                except KeyboardInterrupt:
                    # stop on user request
                    raise KeyboardInterrupt
                except Exception as e:
                    logging.error(traceback.format_exc())
                    continue

                res.extend(ds_res)

            for r in res:
                model_name = r.get("model_UUID", "?")
                model = r.pop("model", None)
                if not model:
                    continue
                storage.save_model(model, model_name)

            storage.save_results(res, str(EXPERIMENT_ID))

        logger.info(f"Done in {main_timer.time}")

    finally:
        logger.removeHandler(fh)
        fh.close()

        storage.save_log(TMP_LOG_FILE_PATH, EXPERIMENT_ID)
