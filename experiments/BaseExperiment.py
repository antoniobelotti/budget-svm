import logging
import math
import os
import pathlib
import pprint
import textwrap
import time
import traceback
import uuid
from abc import abstractmethod, ABC
from collections.abc import Iterator
from typing import Optional

from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from budgetsvm.svm import SVC as our_SVC
from experiments.datasets.Base import Dataset, PrecomputedKernelDataset, PrecomputedKernelDatasetBuilder
from experiments.storage.Storage import Storage
from experiments.utils import Timer


class BaseExperiment(ABC):
    """
    BaseExperiment provides the framework for a generic experiment on SVC models.

    Each experiment works like:
    ```
    foreach dataset ds:
        for each budget_percentage p:
            budget = compute actual budget value
            metadata, model = select_best_model(ds, budget)
            save model in storage
    save all experiment metadata in storage
    ```

    There are two slightly different usages:
    1) use our budgetsvm.svm.SVC model => experiment will use PrecomputedKernels (unless USE_PRECOMPUTED_KERNEL=False)
    2) use another BaseEstimator model => experiment will NOT use precomputedKernels

    TODO: running cvgrid would probably be easier if budgetsvm.svm.SVC.__init__ accepted kernels
            and their parameters like sklearn.svm.SVC (e.g. SVC(kernel="rbf", gamma=1))

    TODO: fix this class: the two different flows for precomputed/not precomputed are ***"!£?^!"?^£!**
    """

    # TODO: config object ?
    USE_PRECOMPUTED_KERNEL: bool = True
    RANDOM_STATE: int = 42
    CROSS_VALIDATION: int
    BUDGET_PERCENTAGES: list[float]
    EXPERIMENT_FILE_SUFFIX: str = ""

    def __init__(self, storage: Storage):
        self.storage = storage

        self.EXPERIMENT_ID = str(time.time()) + self.EXPERIMENT_FILE_SUFFIX
        self.TMP_LOG_FILE_PATH = pathlib.Path("/tmp") / f"{self.EXPERIMENT_ID}.log"

        self.__init_logger()

    def __init_logger(self):
        self.logger = logging.getLogger(f"experiment_{self.EXPERIMENT_ID}")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s[%(levelname)s] \t%(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        fh = logging.FileHandler(self.TMP_LOG_FILE_PATH)
        fh.setFormatter(formatter)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    @abstractmethod
    def generate_datasets(self) -> Iterator[Dataset]:
        pass

    @abstractmethod
    def get_cvgrid_parameter_dict(self) -> dict:
        """
        Return a dict to be passed to sklearn CVGrid object.
        Must contain valid parameters for the model returned by `self.get_empty_model()`
        """
        pass

    @abstractmethod
    def get_empty_model(self):
        """Return a new instance of the model"""
        pass

    def __generate_precomputed_datasets(self, dataset: Dataset) -> Iterator[PrecomputedKernelDataset]:
        """
        Only for budgetdvm.svm.SVC
        Use self.cfg.kernels to generate precomputed kernel datasets
        """
        builder = PrecomputedKernelDatasetBuilder(self.storage, dataset)
        for krn in self.get_cvgrid_parameter_dict().get("kernels"):
            yield builder.build_for(krn)

    def __select_best_model(self, dataset: Dataset | PrecomputedKernelDataset, budget: Optional[int]) -> dict:
        """
        Run CVGrid on a dataset with a budget value. Returns a dict with metadata about the best model and the best
        model itself.
        """

        skf = StratifiedKFold(n_splits=self.CROSS_VALIDATION, shuffle=True, random_state=self.RANDOM_STATE)

        params = self.get_cvgrid_parameter_dict()
        if self.USE_PRECOMPUTED_KERNEL:
            del params["kernels"]
            params["kernel"] = [dataset.kernel]

        params["budget"] = [budget]

        cpus = math.ceil(os.cpu_count() * 0.8)
        cv_grid = GridSearchCV(
            self.get_empty_model(),
            params,
            refit=True,
            verbose=0,
            cv=skf,
            n_jobs=cpus,
        )
        best_model, best_params, best_score = None, None, 0
        try:
            with Timer() as t:
                cv_grid.fit(dataset.X_train, dataset.y_train)
            test_score = cv_grid.score(dataset.X_test, dataset.y_test)
            best_model, best_params, best_score = cv_grid.best_estimator_, cv_grid.best_params_, test_score
        except FitFailedWarning:
            pass
        except Exception:
            self.logger.error("GridSearchCV failed with unexpected error.")
            self.logger.error(traceback.format_exc())

        num_sv = best_model.num_sv_ if hasattr(best_model, "num_sv_") else len(best_model.alpha_)
        res = {
            "dataset": dataset.id if isinstance(dataset, Dataset) else dataset.dataset_id,
            "precomputed_dataset_id": None if isinstance(dataset, Dataset) else dataset.id,
            "model": best_model,
            "model_UUID": uuid.uuid4(),
            "params": best_params,
            "score": best_score,
            "budget": budget,
            "num_sv": num_sv,
            "cv_grid_fit_time": t.time,
        }

        if isinstance(best_model, our_SVC):
            res["solver_status"] = best_model.solver_status_ if hasattr(best_model, "solver_status_") else None
            res["s_pos"] = best_model.s_pos_ if hasattr(best_model, "s_pos_") else None
            res["s_neg"] = best_model.s_neg_ if hasattr(best_model, "s_neg_") else None
            res["b_pos"] = best_model.b_pos_ if hasattr(best_model, "b_pos_") else None
            res["b_neg"] = best_model.b_neg_ if hasattr(best_model, "b_neg_") else None
            res["mip_gap"] = best_model.mip_gap_ if hasattr(best_model, "mip_gap_") else None

        return res

    def __generate_budget_values(self, dataset: Dataset | PrecomputedKernelDataset) -> Iterator[(int, float)]:
        """Generate budget values expressed as percentages of the training dataset."""
        m = len(dataset.X_train)

        dataset_id = dataset.id if isinstance(dataset, Dataset) else dataset.dataset_id

        for p in self.BUDGET_PERCENTAGES:
            budget = int(m * p)
            if budget < 2:
                self.logger.warning(f"{p*100}% of dataset {dataset_id} is too low of a budget")
                continue
            yield int(m * p), p

    def __process_with_precomputed_dataset(self, dataset: Dataset) -> list[dict]:
        results = {}  # stores the best result foreach budget value
        for precomp_dataset in self.__generate_precomputed_datasets(dataset):
            self.logger.debug(f"Processing dataset {precomp_dataset.id}")

            for budget, perc in self.__generate_budget_values(precomp_dataset):
                self.logger.debug(f"budget={perc*100}% of dataset - model selection starting")

                res = self.__select_best_model(precomp_dataset, budget)
                self.logger.info(f"best model achieved {res['score']:.2f} test accuracy with params {res['params']}")
                res["model_name"] = f"{perc:.2f}_budget"

                # fist model being trained with this budget
                if budget not in results:
                    results[budget] = res
                else:
                    # update this budget results and model only if the current model outperforms the previous best one
                    if res.get("score", 0) > results[budget].get("score", 0):
                        results[budget] = res

                self.logger.debug(f"budget={perc*100}% of dataset - model selection done")

        return list(results.values())

    def __process_dataset(self, dataset: Dataset):
        results = []

        self.logger.debug(f"Processing dataset {dataset.id}")

        for budget, perc in self.__generate_budget_values(dataset):
            self.logger.debug(f"budget={perc*100}% of dataset - model selection starting")
            ds_res = self.__select_best_model(dataset, budget)
            self.logger.info(f"best model achieved {ds_res['score']:.2f} test accuracy with params {ds_res['params']}")
            self.logger.debug(f"budget={perc*100}% of dataset - model selection done")

            ds_res["model_name"] = f"{perc:.2f}_budget"

            results.append(ds_res)

        return results

    def run(self):
        if not isinstance(self.get_empty_model(), our_SVC) and self.USE_PRECOMPUTED_KERNEL:
            self.logger.warning(
                f"setting USE_PRECOMPUTED_KERNEL=False - precomputed kernel not supported for this model"
            )
            self.USE_PRECOMPUTED_KERNEL = False

        self.logger.info(
            textwrap.dedent(
                f"""
                Running experiment {self.EXPERIMENT_ID} with the following configuration:
                
                USE_PRECOMPUTED_KERNEL = {self.USE_PRECOMPUTED_KERNEL}
                BUDGET_PERCENTAGES = {self.BUDGET_PERCENTAGES}
                CROSS_VALIDATION = {self.CROSS_VALIDATION}
                RANDOM_STATE = {self.RANDOM_STATE}
                CVGRID_PARAMETERS = {pprint.pformat(self.get_cvgrid_parameter_dict())}
                """
            )
        )

        all_results = []
        for dataset in self.generate_datasets():
            if self.USE_PRECOMPUTED_KERNEL:
                ds_results = self.__process_with_precomputed_dataset(dataset)
            else:
                ds_results = self.__process_dataset(dataset)

            for row in ds_results:
                model = row.pop("model", None)
                model_uuid = row.get("model_UUID", None)

                # save model separately from results
                self.storage.save_model(model, model_uuid)
                all_results.append(row)

        self.storage.save_results(all_results, self.EXPERIMENT_ID)
        self.storage.save_log(self.TMP_LOG_FILE_PATH, self.EXPERIMENT_ID)
