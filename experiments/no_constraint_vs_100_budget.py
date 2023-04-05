from experiments.datasets import generate_datasets
from budgetsvm.svm import SVC
import math
import uuid
from experiments.utils import Timer, model_selection, CustomJSONEncoder
import logging
import time
import pathlib
import os
import textwrap
import pickle
import json

BASE_DIR_PATH = pathlib.Path(os.path.dirname(__file__)).absolute() / pathlib.Path('results')
BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)

NOW = time.time()
OUT_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}_100per_budget.json")
DESCRIPTION_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}_100per_budget.description")


logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR_PATH / f"{NOW}_100per_budget.log"),
            logging.StreamHandler()
            ]
        )

logging.info(
    textwrap.dedent(f"""
        DESCRIPTION:
        
            Foreach dataset:
                train unconstrained SVC
                train with budget==unconstrained model number of SV
           
            - 4 datasets from custom function, 2 dataset from sklearn make_blobs. Each dataset has 300 points.
            - 5-fold stratified cross validation
        """)
)

results = []
for ds in generate_datasets():
    logging.info(f"Starting on dataset {ds.id}")

    with Timer() as t:
        unconstr_model,unconstr_params, unconstr_score = model_selection(SVC(), ds.X_train, ds.X_test, ds.y_train, ds.y_test, cv=5)

    logging.debug(
            f"Dataset {ds.id[-10:]} unconstrained model"
            f"Model selection took {t.time} seconds")

    if not hasattr(unconstr_model, "optimal_"):
        logging.error(f"Trained model {unconstr_model} has no attribute optimal_. This should not happen.")
        unconstr_model.optimal_ = False

    unconstr_model_uuid = uuid.uuid4()
    results.append({
        "dataset": ds.id,
        "model_UUID": unconstr_model_uuid,
        "model": unconstr_model,
        "model_name": "unconstrained",
        "optimal": unconstr_model.optimal_ if unconstr_model else None,
        "params": unconstr_params,
        "obj_fn_value": unconstr_model.obj_,
        "score": unconstr_score,
        "budget": math.inf,
        "num_sv": len(unconstr_model.alpha_) if unconstr_model else None,
        "train_time": t.time
        })

    logging.debug(f"Dataset {ds.id[-10:]} trying 100% budget using {len(unconstr_model.alpha_)} SV")
    with Timer() as t:
        budget_model,budget_params, budget_score = model_selection(SVC(), ds.X_train, ds.X_test, ds.y_train, ds.y_test, cv=5)

    logging.debug(
            f"Dataset {ds.id[-10:]} 100% budget model"
            f"Model selection took {t.time} seconds")

    budget_model_uuid = uuid.uuid4()
    results.append({
        "dataset": ds.id,
        "model_UUID": budget_model_uuid,
        "model": budget_model,
        "model_name": "100perc_budget",
        "optimal": budget_model.optimal_ if budget_model else None,
        "params": budget_params,
        "obj_fn_value": budget_model.obj_,
        "score": budget_score,
        "budget": math.inf,
        "num_sv": len(budget_model.alpha_) if budget_model else None,
        "train_time": t.time
        })


logging.info("Saving models on disk")
pathlib.Path(BASE_DIR_PATH / "models").mkdir(exist_ok=True)
for r in results:
    model_name = r.get("model_UUID", "?")
    model = r.pop("model", None)
    if model is None:
        continue
    with open(BASE_DIR_PATH / "models" / f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

logging.info("Saving results on disk")
with open(OUT_FILE_PATH, "w+") as f:
    f.write(json.dumps(results, cls=CustomJSONEncoder))

logging.info("Done")
