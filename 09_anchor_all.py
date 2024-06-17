import pandas as pd
import numpy as np
import os
import pickle

from collections import namedtuple

from tqdm import tqdm

from alibi.explainers import AnchorTabular

from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def anchor_explainations(data_to_anchor, explainer, model):
    rules = []
    rules_len = []
    prec = []
    cov = []

    for idx in tqdm(range(len(data_to_anchor)), desc=model):
        explanation = explainer.explain(data_to_anchor[idx], threshold=0.95)
        rules.append(" AND ".join(explanation.anchor))
        rules_len.append(len(explanation.anchor))
        prec.append(explanation.precision)
        cov.append(explanation.coverage)

    anchor_exp_dict = {
        "anchor_rule": rules,
        "anchor_rule_len": rules_len,
        "anchor_precision": prec,
        "anchor_coverage": cov,
    }
    return pd.DataFrame(anchor_exp_dict)


def precision(c) -> float:  # pylint: disable=missing-function-docstring
    if (c.p + c.n) == 0:
        return 0
    return c.p / (c.p + c.n)


def coverage(c) -> float:  # pylint: disable=missing-function-docstring
    return c.p / c.P


def correct_precision_coverage(test, train, feature_names, cat_i_bool):
    NCov = namedtuple("NCov", ["p", "n", "P"])
    cov_all_list = []
    prec_all_list = []

    for i in range(len(test)):
        rule_from_anchor = test["anchor_rule"].values[i]
        decision = test["prediction"].values[i]
        conditions_list = rule_from_anchor.split(" AND ")
        filtered_train = train.copy()
        for condition in conditions_list:
            key_value = condition.split(" ")
            if key_value[1] == "=":
                if key_value[0] in np.array(feature_names)[cat_i_bool]:
                    filtered_train = filtered_train[
                        filtered_train[key_value[0]] == float(key_value[2])
                    ]
                else:
                    filtered_train = filtered_train[
                        filtered_train[key_value[0]] == key_value[2]
                    ]
            elif len(key_value) > 3:
                filtered_train = filtered_train.query(condition)
            else:
                if key_value[0] in np.array(feature_names)[cat_i_bool]:
                    filtered_train = filtered_train.query(condition)
                else:
                    filtered_train = filtered_train.query(
                        f"{key_value[0]}{key_value[1]}'{key_value[0]}'"
                    )
        p = len(filtered_train[filtered_train["prediction"] == decision]) + 1
        n = len(filtered_train[filtered_train["prediction"] != decision])
        P = len(train[train["prediction"] == decision]) + 1
        new_cov_loc = NCov(p, n, P)
        cov_all_list.append(coverage(new_cov_loc))
        prec_all_list.append(precision(new_cov_loc))

    return prec_all_list, cov_all_list


def anchor_all(dataset_path, bb_models):
    models = np.unique(bb_models[bb_models["dataset"] == dataset_path]["model"])

    train_df = pd.read_csv(f"results_new/{dataset_path}/train.csv")
    x_train_df = train_df.drop(columns=["target"])
    y_train = train_df["target"].values

    test_df = pd.read_csv(f"results_new/{dataset_path}/test.csv")
    x_test_df = test_df.drop(columns=["target"])
    y_test = test_df["target"].values

    for model_path in models:

        try:

            if not os.path.exists(f"results_anchor_all/{dataset_path}/{model_path}/anchor_train.csv"):    

                with open(f"results_new/{dataset_path}/{model_path}/model.pkl", "rb") as f:
                    model = pickle.load(f)

                    feature_names = list(x_train_df.columns)
                    explainer = AnchorTabular(model.predict_proba, feature_names)
                    explainer.fit(x_train_df.to_numpy(), disc_perc=[25, 50, 75])

                    anchor_train = anchor_explainations(
                        x_train_df.to_numpy(), explainer, f"{dataset_path}/{model_path}"
                    )
                    anchor_test = anchor_explainations(
                        x_test_df.to_numpy(), explainer, f"{dataset_path}/{model_path}"
                    )

                    os.makedirs(
                        f"results_anchor_all/{dataset_path}/{model_path}", exist_ok=True
                    )
                    cat_i_bool = [x_train_df[i].dtype != "O" for i in feature_names]

                    train_pred = pd.read_csv(
                        f"results_all/{dataset_path}/{model_path}/preds_train.csv"
                    )
                    train_all = pd.concat([train_df, train_pred, anchor_train], axis=1)
                    (
                        anchor_precision_correct,
                        anchor_coverage_correct,
                    ) = correct_precision_coverage(
                        train_all, train_all, feature_names, cat_i_bool
                    )
                    train_all["anchor_precision_correct"] = anchor_precision_correct
                    train_all["anchor_coverage_correct"] = anchor_coverage_correct
                    train_all.to_csv(
                        f"results_anchor_all/{dataset_path}/{model_path}/anchor_train.csv",
                        index=False,
                        sep=";",
                    )

                    test_pred = pd.read_csv(
                        f"results_all/{dataset_path}/{model_path}/preds_test.csv"
                    )
                    test_all = pd.concat([test_df, test_pred, anchor_test], axis=1)
                    (
                        anchor_precision_correct,
                        anchor_coverage_correct,
                    ) = correct_precision_coverage(
                        test_all, train_all, feature_names, cat_i_bool
                    )
                    test_all["anchor_precision_correct"] = anchor_precision_correct
                    test_all["anchor_coverage_correct"] = anchor_coverage_correct
                    test_all.to_csv(
                        f"results_anchor_all/{dataset_path}/{model_path}/anchor_test.csv",
                        index=False,
                        sep=";",
                    )
        except:
            print(f"Problem with {dataset_path}/{model_path}")


bb_models = pd.read_csv("results/selected_bb_models.csv")
datasets = bb_models["dataset"].unique()

# Parallel(n_jobs=30)(delayed(anchor_all)(dataset, bb_models) for dataset in datasets)

for dataset in tqdm(datasets):
    anchor_all(dataset, bb_models)
