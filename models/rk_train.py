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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# from xgboost import XGBClassifier

from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.params import Measures
from rulexai.explainer import RuleExplainer

RuleKit.init()


def rk_train_org(dataset_path, X_train, X_test, y_train, y_test, model_type=None):
    rk_dict = {
        "RK_C2": Measures.C2,
        "RK_Corr": Measures.Correlation,
        "RK_Prec": Measures.Precision,
    }

    results_all = pd.DataFrame()

    for mod in rk_dict:
        model_path = mod if model_type is None else f"{model_type}_{mod}"

        os.makedirs(f"results_all/{dataset_path}/{model_path}", exist_ok=True)

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
            f"results_all/{dataset_path}/{model_path}/preds_train_org.csv", index=False
        )
        pd.DataFrame(preds_test, columns=["prediction"]).to_csv(
            f"results_all/{dataset_path}/{model_path}/preds_test_org.csv", index=False
        )

        with open(f"results_all/{dataset_path}/{model_path}/model_org.pkl", "wb") as f:
            pickle.dump(model, f)

        y_train_df = pd.DataFrame(y_train, columns=["class"])
        explainer = RuleExplainer(
            model=model, X=X_train, y=y_train_df.astype(str), type="classification"
        )
        explainer.explain()

        explainer.feature_importances_.to_csv(
            f"results_all/{dataset_path}/{model_path}/fi_class_org.csv", index=False
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

    if model_type is None:
        results_all.to_csv(
            f"results_all/{dataset_path}/rk_models_stats_org.csv", index=False
        )
    else:
        results_all.to_csv(
            f"results_all/{dataset_path}/rk_models_stats_org_{model_type}.csv",
            index=False,
        )
