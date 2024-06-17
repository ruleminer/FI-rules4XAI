import json
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, recall_score, mutual_info_score

from tqdm import tqdm
from rmatrix.classification import RMatrixClassifier
from decision_rules.serialization.utils import JSONSerializer

from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.measures import c2, precision

def statistics_rk_rmatrix(mod, model, y_train, y_test, preds_train, preds_test):
    stats = model.calculate_ruleset_stats()

    results = {
                    'model': mod,
                    'acc_train': accuracy_score(y_train, preds_train),
                    'acc_test': accuracy_score(y_test, preds_test),
                    'bacc_train': balanced_accuracy_score(y_train, preds_train),
                    'bacc_test': balanced_accuracy_score(y_test, preds_test),
                    'kappa_train': cohen_kappa_score(y_train, preds_train),
                    'kappa_test': cohen_kappa_score(y_test, preds_test),
                    "mutual_train": mutual_info_score(y_train, preds_train),
                    "mutual_test": mutual_info_score(y_test, preds_test),
                    'recall_train': recall_score(y_train, preds_train, average="macro"),
                    'recall_test': recall_score(y_test, preds_test, average="macro"),
                    'avg_rule_coverage': stats["avg_coverage"],
                    'avg_rule_precision': stats["avg_precision"],
                    'avg_conditions_count': stats["avg_conditions_count"],
                    'rules_count': stats["rules_count"]
                }
    return results

bb_models = pd.read_csv("results/selected_bb_models.csv")
datasets = bb_models["dataset"].unique()

class_types = ["_filterFF_precision_approx", "_filterFF_precision_global", "_filterFF_precision_local"]

for class_type in class_types:

    for sel_dataset in tqdm(datasets, desc="Datasets"):

        models = np.unique(bb_models[bb_models["dataset"]==sel_dataset]["model"])

        x_train_df = pd.read_csv(f"../results_all/{sel_dataset}/train.csv")
        x_train_df.drop(columns=["target"], inplace=True)

        x_test_df = pd.read_csv(f"../results_all/{sel_dataset}/test.csv")
        x_test_df.drop(columns=["target"], inplace=True)

        binary_columns = list(x_train_df.columns[x_train_df.isin([0,1]).all()])
        if len(binary_columns) > 0:
            x_train_df[binary_columns] = x_train_df[binary_columns].astype(str)
            x_test_df[binary_columns] = x_test_df[binary_columns].astype(str)

        results_all = pd.DataFrame()

        for sel_model in models:

            y_train_df = pd.read_csv(f"../results_all/{sel_dataset}/{sel_model}/preds_train.csv")
            y_train_df = y_train_df.rename(columns={'prediction': 'name'})
            y_train = y_train_df["name"].squeeze().astype(str)

            y_test_df = pd.read_csv(f"../results_all/{sel_dataset}/{sel_model}/preds_test.csv")
            y_test_df = y_test_df.rename(columns={'prediction': 'name'})
            y_test = y_test_df["name"].squeeze().astype(str)

            try: 

                file_path= f"../results_all/{sel_dataset}/{sel_model}/ruleset{class_type}.json"
                with open(file_path, 'r') as json_file:
                    ruleset_json_read = json.load(json_file)

                classifier = JSONSerializer.deserialize(ruleset_json_read, target_class=ClassificationRuleSet)
                if "c2" in class_type:
                    classifier.update(x_train_df, y_train, measure=c2)
                elif "precision" in class_type:
                    classifier.update(x_train_df, y_train, measure=precision)

                preds_train = classifier.predict(x_train_df)
                preds_test = classifier.predict(x_test_df)

                results = statistics_rk_rmatrix(sel_model, classifier, y_train, y_test, preds_train, preds_test)
                results_df = pd.DataFrame(results, index=[0])
                results_all = pd.concat([results_all, results_df])

                results_all.to_csv(f"../results_all/{sel_dataset}/rmatrix_models_stats{class_type}.csv", index=False)

            except:

                print(f"../results_all/{sel_dataset}/{sel_model}/ruleset{class_type}.json")

for class_type in class_types:

    rmatrix_stats_all = pd.DataFrame()

    for sel_dataset in tqdm(datasets, desc="Datasets"):

        try: 

            results = pd.read_csv(f"../results_all/{sel_dataset}/rmatrix_models_stats{class_type}.csv")
            results["dataset"] = sel_dataset
            rmatrix_stats_all = pd.concat([rmatrix_stats_all, results])

        except:
            print(f"../results_all/{sel_dataset}/rmatrix_models_stats{class_type}.csv")

    rmatrix_stats_all.to_csv(f"../results/rmatrix_models_stats{class_type}.csv", index=False)
