import pandas as pd

from experiments.storage.GDriveStorage import GDriveStorage
from budgetsvm.kernel import *
from budgetsvm.svm import SVC


storage = GDriveStorage()

df = pd.concat([
  storage.get_result_dataframe("1688809754.2443595.json"),
  storage.get_result_dataframe("1688975220.8464673.json"),
  storage.get_result_dataframe("1689153514.3068242.json")
], ignore_index=True)

prev_results = df.query("solver_status == 9")

res = []
from tqdm import tqdm

for i, row in tqdm(prev_results.iterrows()):
    try:
        kernel = eval(row["params"]["kernel"])
        if kernel.precomputed:
            kernel = kernel.original_kernel
        C = row["params"]["C"]
        model = SVC(kernel=kernel, C=C, budget=int(float(row["budget"])))
        ds = storage.get_dataset_if_exists(row["dataset"])
        model.fit(ds.X_train, ds.y_train)

        res.append({
            "old_df_idx": i,
            "solver_status_": model.solver_status_,
            "mip_gap_": model.mip_gap_
        })
    except:
        print("nvkedblakuejnm")
        pass
    break

storage.save_results(res, "correct_mipgap_exp2d")