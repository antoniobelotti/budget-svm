import json
import logging
import math
import os
import pathlib
import textwrap
import time
import uuid

from sklearn.model_selection import ParameterGrid

from budgetsvm.svm import SVC
from experiments.main import get_datasets
from experiments.storage.GDriveStorage import GDriveStorage
from experiments.utils import Timer, CustomJSONEncoder
from kernel import LinearKernel, GaussianKernel, PolynomialKernel
from optimization import GurobiSolver

CWD = pathlib.Path(os.path.dirname(__file__)).absolute()
EXPERIMENT_ID = str(time.time()) + "_100per_budget_no_cv"
TMP_LOG_FILE_PATH = CWD / f"{EXPERIMENT_ID}.log"

with open(pathlib.Path(CWD / "100_budget_config.json"), "rb") as f:
    config = json.load(f)

logger = logging.getLogger("experiments")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s[%(levelname)s] \t%(message)s')
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

storage = GDriveStorage()

results = []
for ds in get_datasets(config["datasets"], storage):
    logger.info(f"Starting on dataset {ds.id}")

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

    grid_params = ParameterGrid({"C": c_values, "kernel": kernel_values})
    for params in grid_params:
        logger.debug(f"Dataset {ds.id[-10:]} params {params} ")
        logger.debug("training unconstrained model")
        unconstrained_model_solver = GurobiSolver()
        with Timer() as t:
            unconstrained_model = SVC(**params)
            try:
                unconstrained_model.fit(
                    ds.X_train, ds.y_train, solver=unconstrained_model_solver, keep_alpha_equal_C=True
                )
                score = unconstrained_model.score(ds.X_test, ds.y_test)
            except:
                unconstrained_model = None
                score = 0

        if unconstrained_model is None:
            continue

        unconstrained_model_UUID = uuid.uuid4()
        storage.save_model(unconstrained_model, str(unconstrained_model_UUID))
        results.append(
            {
                "dataset": ds.id,
                "model_name": "unconstrained",
                "model_UUID": unconstrained_model_UUID,
                "params": {k: str(v) for k, v in params.items()},
                "score": score,
                "obj_fn_value": unconstrained_model.obj_ if unconstrained_model else 0,
                "num_sv": len(unconstrained_model.alpha_) if unconstrained_model else 0,
                "budget": math.inf,
                "solver_status": unconstrained_model.solver_status_
                if unconstrained_model
                else False,
                "train_time": t.time,
                "s_pos": unconstrained_model.s_pos_ if unconstrained_model else 0,
                "s_neg": unconstrained_model.s_neg_ if unconstrained_model else 0,
                "b_pos": unconstrained_model.b_pos_ if unconstrained_model else 0,
                "b_neg": unconstrained_model.b_neg_ if unconstrained_model else 0,
                "mip_gap": unconstrained_model.mip_gap_
            }
        )
        logger.debug(f"unconstrained model has {len(unconstrained_model.alpha_)} SV")
        logger.debug(f"training model with budget={len(unconstrained_model.alpha_)}")
        budget = len(unconstrained_model.alpha_)

        budgeted_model_solver = GurobiSolver()
        with Timer() as t:
            budgeted_model = SVC(budget=budget, **params)
            try:
                budgeted_model.fit(ds.X_train, ds.y_train, solver=budgeted_model_solver, keep_alpha_equal_C=True)
                score = budgeted_model.score(ds.X_test, ds.y_test)
            except:
                budgeted_model = None
                score = 0

        budgeted_model_UUID = uuid.uuid4()
        storage.save_model(budgeted_model, str(budgeted_model_UUID))
        results.append(
            {
                "dataset": ds.id,
                "model_name": "100perc_budget",
                "model_UUID": budgeted_model_UUID,
                "params": {k: str(v) for k, v in params.items()},
                "score": score,
                "obj_fn_value": budgeted_model.obj_ if budgeted_model else 0,
                "num_sv": len(budgeted_model.alpha_) if budgeted_model else 0,
                "budget": budget,
                "solver_status": budgeted_model.solver_status_ if budgeted_model else False,
                "train_time": t.time,
                "s_pos": budgeted_model.s_pos_ if budgeted_model else 0,
                "s_neg": budgeted_model.s_neg_ if budgeted_model else 0,
                "b_pos": budgeted_model.b_pos_ if budgeted_model else 0,
                "b_neg": budgeted_model.b_neg_ if budgeted_model else 0,
                "mip_gap": budgeted_model.mip_gap_
            }
        )
        logger.debug(f"done for dataset {ds.id[-10:]}")
        logger.debug(f"unconstrained model has {len(budgeted_model.alpha_)} SV")

        if (
            unconstrained_model
            and budgeted_model
            and unconstrained_model.optimal_
            and budgeted_model.optimal_
            and unconstrained_model.obj_ != budgeted_model.obj_
        ):
            logger.error(
                f"both optimal, different obj fun value. models:[\n\tunconstr: {unconstrained_model_UUID}\n\tbudgeted: {budgeted_model_UUID}\n]"
            )
            # TODO: use storage
            #unconstrained_model_solver.model.write(f"{unconstrained_model_UUID}.lp")
            #budgeted_model_solver.model.write(f"{budgeted_model_UUID}.lp")

storage.save_results(results, EXPERIMENT_ID)

logger.info("Done")


logger.removeHandler(fh)
fh.close()

storage.save_log(TMP_LOG_FILE_PATH, EXPERIMENT_ID)