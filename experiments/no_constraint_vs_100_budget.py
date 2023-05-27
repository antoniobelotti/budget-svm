import json
import logging
import math
import os
import pathlib
import textwrap
import time

from sklearn.model_selection import ParameterGrid

from budgetsvm.svm import SVC
from experiments.main import get_datasets
from experiments.utils import Timer, CustomJSONEncoder
from kernel import LinearKernel, GaussianKernel, PolynomialKernel

CWD = pathlib.Path(os.path.dirname(__file__)).absolute()
BASE_DIR_PATH = pathlib.Path(CWD / "results")
BASE_DIR_PATH.mkdir(parents=True, exist_ok=True)

NOW = time.time()
OUT_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}_100per_budget_no_cv.json")
DESCRIPTION_FILE_PATH = pathlib.Path(BASE_DIR_PATH / f"{NOW}_100per_budget_no_cv.description")

with open(pathlib.Path(CWD / "100_budget_config.json"), "rb") as f:
    config = json.load(f)

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR_PATH / f"{NOW}_100per_budget_no_cv.log"),
            logging.StreamHandler()
            ]
        )

logging.info(
    textwrap.dedent(
        f"""
            Running experiment with the following configuration:

            {json.dumps(config, indent=4)}
            """
    )
)

results = []
for ds in get_datasets(config["datasets"]):
    logging.info(f"Starting on dataset {ds.id}")

    c_values = config["model_selection"].get("C", [1.0])

    kernel_values = []
    for kernel_name, hp in config["model_selection"].get("kernels", ["linear"]).items():
        if kernel_name == "linear":
            kernel_values.append(LinearKernel())
        if kernel_name == "gaussian":
            for v in hp:
                kernel_values.append(GaussianKernel(v))
        if kernel_name == "polynomial":
            for v in hp:
                kernel_values.append(PolynomialKernel(v))

    grid_params = ParameterGrid({'C': c_values, 'kernel': kernel_values})
    for params in grid_params:
        logging.debug(f"Dataset {ds.id[-10:]} params {params} ")
        logging.debug("training unconstrained model")
        with Timer() as t:
            model = SVC(**params)
            try:
                model.fit(ds.X_train, ds.y_train)
                score = model.score(ds.X_test, ds.y_test)
            except:
                model=None
                score = 0

        if model is None:
            continue

        results.append({
            "dataset": ds.id,
            "model_name": "unconstrained",
            "params": {k:str(v) for k,v in params.items()},
            "score": score,
            "obj_fn_value": model.obj_ if model else 0,
            "num_sv": len(model.alpha_) if model else 0,
            "budget": math.inf,
            "optimal": model.optimal_ if model else False,
            "train_time": t.time
            })
        logging.debug(f"unconstrained model has {len(model.alpha_)} SV")
        logging.debug(f"training model with budget={len(model.alpha_)}")
        budget = len(model.alpha_)

        with Timer() as t:
            model = SVC(budget=budget, **params)
            try:
                model.fit(ds.X_train, ds.y_train)
                score = model.score(ds.X_test, ds.y_test)
            except:
                model=None
                score = 0

        results.append({
            "dataset": ds.id,
            "model_name": "100perc_budget",
            "params": {k:str(v) for k,v in params.items()},
            "score": score,
            "obj_fn_value": model.obj_ if model else 0,
            "num_sv": len(model.alpha_) if model else 0,
            "budget": budget,
            "optimal": model.optimal_ if model else False,
            "train_time": t.time
        })
        logging.debug(f"done for dataset {ds.id[-10:]}")


logging.info("Saving results on disk")
with open(OUT_FILE_PATH, "w+") as f:
    f.write(json.dumps(results, cls=CustomJSONEncoder))

logging.info("Done")
