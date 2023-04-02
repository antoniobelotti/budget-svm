""" to suppress sklearn warnings. Many warning are thrown during model selection."""


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import logging
import datetime
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
from experiments.datasets import generate_datasets
from experiments.utils import Timer, model_selection, CustomJSONEncoder


def task_experiment_on_dataset(dataset):
    best_sv_number = None
    prev_budget = None

    results = []
    for perc in np.linspace(1.0, 0.3, 8):
        budget = None
        if best_sv_number:
            budget = int(best_sv_number * perc)
            if budget == prev_budget or budget < 2:
                logging.debug(
                    f"dataset {dataset.id}: skip budget {perc * 100} either because it's <2 or equal to previous "
                    f"iteration budget")
            prev_budget = budget

        model = SVC(budget=budget) if budget else SVC()
        model_name = "full_budget" if not best_sv_number else f"{perc:.2f}_budget"

        logging.debug(f"Dataset {dataset.id[-10:]} Budget {perc * 100}% - Launching model selection")

        with Timer() as t:
            best_model, params, score = model_selection(model, dataset.X_train, dataset.X_test, dataset.y_train,
                                                        dataset.y_test, cv=5)
        logging.debug(
            f"Dataset {dataset.id[-10:]} Budget {perc * 100}% - "
            f"Model selection took {t.time} seconds")

        best_model_uuid = uuid.uuid4()
        results.append({
            "dataset": dataset.id,
            "model_UUID": best_model_uuid,
            "model": best_model,
            "model_name": model_name,
            "optimal": best_model.optimal_ if best_model else None,
            "params": params,
            "score": score,
            "budget": budget if budget else math.inf,
            "num_sv": len(best_model.alpha_) if best_model else None,
            "train_time": t.time
        })

        # if current trained model is full budget save the number of sv.
        if not best_sv_number:
            best_sv_number = len(best_model.alpha_)

    return results


if __name__ == "__main__":
    BASE_DIR_PATH = pathlib.Path(os.path.dirname(__file__)).absolute() / pathlib.Path('results')
    BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)

    NOW = time.time()
    OUT_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}.json")
    DESCRIPTION_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}.description")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR_PATH / f"{NOW}.log"),
            logging.StreamHandler()
        ]
    )

    np.random.seed(42)

    logging.info(
        textwrap.dedent(f"""
            DESCRIPTION:
            
                Foreach dataset:
                    train unconstrained SVC
                    For budget in [.3,.4,.5,.6,.7,.8,.9]% of sup.vec. of unconstrained model
                        train SVC(budget=budget)
               
                - 4 datasets from custom function, 2 dataset from sklearn make_blobs. Each dataset has 300 points.
                - 5-fold stratified cross validation
            """)
    )

    res = []
    for ds in generate_datasets():
        logging.info(f"Launching experiments on dataset f{ds.id}")
        res.extend(task_experiment_on_dataset(ds))

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

    logging.info("Done")
