import pandas as pd
import numpy as np
import os
import copy

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from models.bb_train import bb_approx, rk_train

bb_models = pd.read_csv("results/selected_bb_models.csv")
datasets = bb_models["dataset"].unique()

for dataset_path in tqdm(datasets):
    df_full = pd.read_csv(f"data_csv/{dataset_path}.csv")

    if dataset_path == "nursery":
        df_full = df_full[df_full["class"] != "recommend"]

    y_data = df_full[["class"]]
    ord_tr = OrdinalEncoder()
    y = ord_tr.fit_transform(y_data)
    y = np.array([int(obs[0]) for obs in y])

    x_data = df_full.drop(columns=["class"])
    cols_num = x_data.select_dtypes(include="number").columns.tolist()
    cols_str = x_data.select_dtypes(include="object").columns.tolist()
    x = pd.get_dummies(x_data, columns=cols_str) if len(cols_str) > 0 else x_data
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    X = pd.DataFrame(imp.fit_transform(x), columns=x.columns)

    os.makedirs(f"results_all/{dataset_path}", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    data_train = copy.deepcopy(X_train)
    data_train["target"] = y_train

    data_test = copy.deepcopy(X_test)
    data_test["target"] = y_test

    data_train.to_csv(f"results_all/{dataset_path}/train.csv", index=False)
    data_test.to_csv(f"results_all/{dataset_path}/test.csv", index=False)

    bb_approx(dataset_path, X_train, X_test, y_train, y_test)
    rk_train(dataset_path, X_train, X_test, y_train, y_test)
