import datetime
import itertools
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import textwrap
import time
import uuid
from multiprocessing import Pool

import numpy as np

from budgetsvm.svm import SVC
from experiments.datasets import dataset_from_custom_function, generate_unbalanced_datasets
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
                continue
            prev_budget = budget

        model = SVC(budget=budget) if budget else SVC()
        model_name = "full_budget" if not best_sv_number else f"{perc:.2f}_budget"

        with Timer() as t:
            best_model, params, score = model_selection(model, dataset.X_train, dataset.X_test, dataset.y_train,
                                                        dataset.y_test, cv=5)
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

    np.random.seed(42)

    with open(DESCRIPTION_FILE_PATH, "w") as f:
        f.write(
            textwrap.dedent(f"""
                Launched {datetime.timedelta(seconds=NOW)}.

                Foreach dataset:
                    train unconstrained SVC
                    For budget in [.3,.4,.5,.6,.7,.8,.9]% of sup.vec. of unconstrained model
                        train SVC(budget=budget)
               
                each dataset has 300 points. Can be unbalanced. 
                5-fold stratified cross validation
            """)
        )

    NUM_CORES = multiprocessing.cpu_count()
    POOL_SIZE = int(NUM_CORES / 2)

    datasets = list(generate_unbalanced_datasets())

    with Pool() as p:
        results = p.map(task_experiment_on_dataset, datasets)

    # flatten results
    results = list(itertools.chain.from_iterable(results))

    pathlib.Path(BASE_DIR_PATH / "models").mkdir(exist_ok=True)
    for r in results:
        model_name = r.get("model_UUID", "?")
        model = r.pop("model", None)
        if model is None:
            continue
        with open(BASE_DIR_PATH / "models" / f"{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(OUT_FILE_PATH, "w+") as f:
        f.write(json.dumps(results, cls=CustomJSONEncoder))
