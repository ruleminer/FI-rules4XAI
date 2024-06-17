import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    recall_score,
    accuracy_score,
    mutual_info_score,
)
from xgboost import XGBClassifier

from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.params import Measures
from rulexai.explainer import RuleExplainer

RuleKit.init()

# BB models and RK explanation

def bb_approx(dataset_path, X_train, X_test, y_train, y_test, models):

    model_names = [
        f"{model}_{str(i)}" for model in ["XGB", "RF", "SVC"] for i in range(3)
    ]

    params = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.2, "random_state": 42},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.3, "random_state": 42},
        {"n_estimators": 100, "criterion": "gini", "random_state": 42},
        {"n_estimators": 200, "criterion": "gini", "random_state": 42},
        {"n_estimators": 100, "criterion": "entropy", "random_state": 42},
        {"kernel": "rbf", "C": 1.0, "probability": True},
        {"kernel": "linear", "C": 1.0, "probability": True},
        {"gamma": "auto", "C": 1.0, "probability": True},
    ]

    rk_dict = {
        "RK_C2": Measures.C2,
        "RK_Corr": Measures.Correlation,
        "RK_Prec": Measures.Precision,
    }

    results_all = pd.DataFrame()
    results_all_approx = pd.DataFrame()

    # for pars, mod in enumerate(model_names):
    #     if pars > 6 and dataset_path in ["credit-a", "madelon", "seismic-bumps"]:
    #         break

    #     print(mod)
    #     os.makedirs(f"results_all/{dataset_path}/{mod}", exist_ok=True)
    #     if mod.split("_")[0] == "XGB":
    #         model = XGBClassifier()
    #     elif mod.split("_")[0] == "RF":
    #         model = RandomForestClassifier()
    #     elif mod.split("_")[0] == "SVC":
    #         model = SVC()

    #     model.set_params(**params[pars])

    #     model.fit(X_train, y_train)
    #     preds_train = model.predict(X_train)
    #     preds_test = model.predict(X_test)

    #     pd.DataFrame(preds_train, columns=["prediction"]).to_csv(
    #         f"results_all/{dataset_path}/{mod}/preds_train.csv", index=False
    #     )
    #     pd.DataFrame(preds_test, columns=["prediction"]).to_csv(
    #         f"results_all/{dataset_path}/{mod}/preds_test.csv", index=False
    #     )

    #     with open(f"results_all/{dataset_path}/{mod}/model.pkl", "wb") as f:
    #         pickle.dump(model, f)

    #     # feature importance overall
    #     fi = permutation_importance(
    #         model, X_train, y_train, n_repeats=10, random_state=42
    #     ).importances_mean
    #     fi_dict = {"attribute": X_train.columns, "importance": fi}

    #     pd.DataFrame(fi_dict).to_csv(
    #         f"results_all/{dataset_path}/{mod}/fi.csv", index=False
    #     )

        # feature importance per class
        # fi_c_all = []

        # for c in set(y_train):
        #     fi_c = permutation_importance(model, X_train[y_train == c], y_train[y_train == c], n_repeats=10, random_state=42).importances_mean
        #     fi_c_all.append(fi_c)

        # columns_names = [f"class_{c}" for c in set(y_train)]
        # fi_c_all_t = np.array(fi_c_all).T.tolist()
        # fi_c_df = pd.DataFrame(fi_c_all_t, columns=columns_names)
        # fi_c_df["attribute"] = X_train.columns

        # fi_c_df.to_csv(f"results_all/{dataset_path}/{mod}/fi_class.csv", index=False)

    for mod in models:

        y_train_df = pd.read_csv(f"results_all/{dataset_path}/{mod}/preds_train.csv")
        preds_train = y_train_df["prediction"].values
        y_test_df = pd.read_csv(f"results_all/{dataset_path}/{mod}/preds_test.csv")
        preds_test = y_test_df["prediction"].values

        results = {
            "model": mod,
            "acc_train": accuracy_score(y_train, preds_train),
            "acc_test": accuracy_score(y_test, preds_test),
            "bacc_train": balanced_accuracy_score(y_train, preds_train),
            "bacc_test": balanced_accuracy_score(y_test, preds_test),
            "kappa_train": cohen_kappa_score(y_train, preds_train),
            "kappa_test": cohen_kappa_score(y_test, preds_test),
            "mutual_train": mutual_info_score(y_train, preds_train),
            "mutual_test": mutual_info_score(y_test, preds_test),
            "recall_train": recall_score(y_train, preds_train, average="macro"),
            "recall_test": recall_score(y_test, preds_test, average="macro"),
        }

        results_df = pd.DataFrame(results, index=[0])
        results_all = pd.concat([results_all, results_df])

        # approximation with rules
        for approx in rk_dict:
            approx_name = f"{mod}_{approx}"

            os.makedirs(f"results_all/{dataset_path}/{approx_name}", exist_ok=True)

            model = RuleClassifier(
                induction_measure=rk_dict[approx],
                pruning_measure=rk_dict[approx],
                voting_measure=rk_dict[approx],
                min_rule_covered=3,
                max_uncovered_fraction=0,
                select_best_candidate=True
            )
            model.fit(X_train, preds_train)
            preds_train_df = pd.DataFrame(preds_train, columns=["class"])
            approx_train, classification_metrics = model.predict(
                X_train, return_metrics=True
            )
            approx_test, classification_metrics = model.predict(
                X_test, return_metrics=True
            )

            approx_train = [int(p) for p in approx_train]
            approx_test = [int(p) for p in approx_test]

            pd.DataFrame(approx_train, columns=["prediction"]).to_csv(
                f"results_all/{dataset_path}/{approx_name}/approx_train.csv",
                index=False,
            )
            pd.DataFrame(approx_test, columns=["prediction"]).to_csv(
                f"results_all/{dataset_path}/{approx_name}/approx_test.csv", index=False
            )

            explainer = RuleExplainer(
                model=model,
                X=X_train,
                y=preds_train_df.astype(str),
                type="classification",
            )
            explainer.explain()

            explainer.feature_importances_.to_csv(
                f"results_all/{dataset_path}/{approx_name}/fi_class.csv", index=False
            )

            with open(f"results_all/{dataset_path}/{approx_name}/model.pkl", "wb") as f:
                pickle.dump(model, f)

            results = {
                "model": approx_name,
                "acc_train": accuracy_score(y_train, approx_train),
                "acc_test": accuracy_score(y_test, approx_test),
                "bacc_train": balanced_accuracy_score(y_train, approx_train),
                "bacc_test": balanced_accuracy_score(y_test, approx_test),
                "kappa_train": cohen_kappa_score(y_train, approx_train),
                "kappa_test": cohen_kappa_score(y_test, approx_test),
                "mutual_train": mutual_info_score(y_train, preds_train),
                "mutual_test": mutual_info_score(y_test, preds_test),
                "recall_train": recall_score(y_train, approx_train, average="macro"),
                "recall_test": recall_score(y_test, approx_test, average="macro"),
                "avg_rule_coverage": model.model.stats.avg_rule_coverage,
                "avg_rule_precision": model.model.stats.avg_rule_precision,
                "avg_rule_quality": model.model.stats.avg_rule_quality,
                "rules_count": model.model.stats.rules_count,
            }

            results_df_approx = pd.DataFrame(results, index=[0])
            results_all_approx = pd.concat([results_all_approx, results_df_approx])

    results_all_approx.to_csv(
        f"results_all/{dataset_path}/rk_approx_stats.csv", index=False
    )
    results_all.to_csv(f"results_all/{dataset_path}/bb_models_stats.csv", index=False)


def rk_train(dataset_path, X_train, X_test, y_train, y_test):
    rk_dict = {
        "RK_C2": Measures.C2,
        "RK_Corr": Measures.Correlation,
        "RK_Prec": Measures.Precision,
    }

    results_all = pd.DataFrame()

    for mod in rk_dict:
        os.makedirs(f"results_all/{dataset_path}/{mod}", exist_ok=True)

        model = RuleClassifier(
            induction_measure=rk_dict[mod],
            pruning_measure=rk_dict[mod],
            voting_measure=rk_dict[mod],
            min_rule_covered=3,
            max_uncovered_fraction=0,
            select_best_candidate=True
        )
        model.fit(X_train, y_train)
        preds_train, classification_metrics = model.predict(
            X_train, return_metrics=True
        )
        preds_test, classification_metrics = model.predict(X_test, return_metrics=True)

        preds_train = [int(p) for p in preds_train]
        preds_test = [int(p) for p in preds_test]

        pd.DataFrame(preds_train, columns=["prediction"]).to_csv(
            f"results_all/{dataset_path}/{mod}/preds_train.csv", index=False
        )
        pd.DataFrame(preds_test, columns=["prediction"]).to_csv(
            f"results_all/{dataset_path}/{mod}/preds_test.csv", index=False
        )

        with open(f"results_all/{dataset_path}/{mod}/model.pkl", "wb") as f:
            pickle.dump(model, f)

        y_train_df = pd.DataFrame(y_train, columns=["class"])
        explainer = RuleExplainer(
            model=model, X=X_train, y=y_train_df.astype(str), type="classification"
        )
        explainer.explain()

        explainer.feature_importances_.to_csv(
            f"results_all/{dataset_path}/{mod}/fi_class.csv", index=False
        )

        results = {
            "model": mod,
            "acc_train": accuracy_score(y_train, preds_train),
            "acc_test": accuracy_score(y_test, preds_test),
            "bacc_train": balanced_accuracy_score(y_train, preds_train),
            "bacc_test": balanced_accuracy_score(y_test, preds_test),
            "kappa_train": cohen_kappa_score(y_train, preds_train),
            "kappa_test": cohen_kappa_score(y_test, preds_test),
            "mutual_train": mutual_info_score(y_train, preds_train),
            "mutual_test": mutual_info_score(y_test, preds_test),
            "recall_train": recall_score(y_train, preds_train, average="macro"),
            "recall_test": recall_score(y_test, preds_test, average="macro"),
            "avg_rule_coverage": model.model.stats.avg_rule_coverage,
            "avg_rule_precision": model.model.stats.avg_rule_precision,
            "avg_rule_quality": model.model.stats.avg_rule_quality,
            "rules_count": model.model.stats.rules_count,
        }

        results_df = pd.DataFrame(results, index=[0])
        results_all = pd.concat([results_all, results_df])

    results_all.to_csv(f"results_all/{dataset_path}/rk_models_stats.csv", index=False)
