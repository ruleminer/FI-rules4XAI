import pandas as pd
import numpy as np
import os
import pickle

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance

from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.params import Measures
from rulexai.explainer import RuleExplainer

RuleKit.init()

def rk_fi_ci(dataset_path, X_train, y_train, model_type=None, suffix=""):

    rk_dict = {'RK_C2': Measures.C2,
               'RK_Corr': Measures.Correlation,
                "RK_Prec": Measures.Precision}

    for mod in rk_dict:

        model_path = mod if model_type is None else f"{model_type}_{mod}"

        with open(f"results_all/{dataset_path}/{mod}/model{suffix}.pkl", 'rb') as f:
            model = pickle.load(f)
     
        y_train_df = pd.DataFrame(y_train, columns=["class"])
        explainer = RuleExplainer(model=model, X=X_train, y=y_train_df.astype(str), type="classification")
        explainer.explain()

        explainer.feature_importances_.to_csv(f"results_all/{dataset_path}/{model_path}/fi_class{suffix}.csv", index=False)
        explainer.condition_importances_.to_csv(f"results_all/{dataset_path}/{model_path}/ci_class{suffix}.csv", index=False)


def prepare_original_data(sel_dataset):
    d = pd.read_csv(f"data_csv/{sel_dataset}.csv")

    if sel_dataset == "nursery":
        d = d[d["class"]!="recommend"]

    y_data = d[["class"]]
    ord_tr = OrdinalEncoder()
    y = ord_tr.fit_transform(y_data)
    y = np.array([int(obs[0]) for obs in y])
    X = d.drop(columns=["class"])
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def prepare_dummy_data(sel_dataset):

    df_full = pd.read_csv(f"data_csv/{sel_dataset}.csv")

    if sel_dataset == "nursery":
        df_full = df_full[df_full["class"]!="recommend"]

    y_data = df_full[["class"]]
    ord_tr = OrdinalEncoder()
    y = ord_tr.fit_transform(y_data)
    y = np.array([int(obs[0]) for obs in y])

    x_data = df_full.drop(columns=["class"])
    cols_str = x_data.select_dtypes(include="object").columns.tolist()
    x = pd.get_dummies(x_data, columns = cols_str) if len(cols_str) > 0 else x_data
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X = pd.DataFrame(imp.fit_transform(x), columns=x.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def calculate_feature_importance_m0(dataset_path, X_train, y_train, model_type=None, suffix=""):

    rk_dict = {'RK_C2': Measures.C2,
               "RK_Prec": Measures.Precision}

    for mod in rk_dict:

        model_path = mod if model_type is None else f"{model_type}_{mod}"

        with open(f"results_all/{dataset_path}/{model_path}/model{suffix}.pkl", 'rb') as f:
            model = pickle.load(f)

        fi = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42).importances_mean
        fi_dict = {'attribute': X_train.columns,
                   'importance': fi}
        fi_df = pd.DataFrame(fi_dict)

        fi_df.to_csv(f"results_all/{dataset_path}/{model_path}/fi{suffix}.csv", index=False)


def calculate_feature_importance_m1(dataset_path, model_type=None, suffix=""):

    rk_dict = {'RK_C2': Measures.C2,
               'RK_Corr': Measures.Correlation,
               "RK_Prec": Measures.Precision}

    for mod in rk_dict:
    
        model_path = mod if model_type is None else f"{model_type}_{mod}"
        
        with open(f"results_all/{dataset_path}/{model_path}/model{suffix}.pkl", 'rb') as f:
                    model = pickle.load(f)

        test_df = pd.read_csv(f"results_all/{dataset_path}/test.csv")
        dataset_attrs = test_df.columns[:-1]
        n_cond = np.sum([col in str(rule) for col in dataset_attrs for rule in model.model.rules])

        n_cond_col = []

        for col in dataset_attrs:
            n_cond_col.append(np.sum([col in str(rule) for rule in model.model.rules]))

        attr_fi = n_cond_col/n_cond

        fi_dict = {'attribute': dataset_attrs, 'importance': attr_fi}
        fi_df = pd.DataFrame(fi_dict)

        fi_df.to_csv(f"results_all/{dataset_path}/{model_path}/fi_method1{suffix}.csv", index=False)


def calculate_feature_importance_m3(dataset_path, model_type=None, suffix=""):
     
    rk_dict = {'RK_C2': Measures.C2,
               'RK_Corr': Measures.Correlation,
               "RK_Prec": Measures.Precision}
    
    test_df = pd.read_csv(f"results_all/{dataset_path}/test.csv")
    dataset_attrs = test_df.columns[:-1]

    for mod in rk_dict:
    
        model_path = mod if model_type is None else f"{model_type}_{mod}"

        ci = pd.read_csv(f"results_all/{dataset_path}/{model_path}/ci_class{suffix}.csv")

        data_columns = 2
        ci_columns = list(ci.columns)
        iters = int(len(ci_columns)/data_columns)

        ci_long = pd.DataFrame()

        num = 0
        for i in range(iters):

            cols_to_select = [0+num,1+num]
            ci_cols = ci.iloc[:, cols_to_select]
            ci_cols.columns = ["condition", "importance"]
            ci_cols = ci_cols[ci_cols["condition"] != "-"]
            ci_long = pd.concat([ci_long, ci_cols])
            num = data_columns

        ci_long["attribute"] = [row.split('=')[0].strip() for row in ci_long['condition']]
        ci_long["importance"] = ci_long["importance"].astype(float)
        fi = ci_long.groupby(["attribute"]).sum(['importance']).reset_index()
        fi["importance_relative"] = fi["importance"]/np.sum(ci_long["importance"])

        not_used_attrs = [x for x in dataset_attrs if x not in fi["attribute"].values]
        attrs_dict = {'attribute': not_used_attrs}
        attrs_df = pd.DataFrame(attrs_dict)
        attrs_df["importance"] = -1000
        attrs_df["importance_relative"] = -1000

        fi = pd.concat([fi, attrs_df])
        # fi["importance_scaled"] = (fi["importance_relative"] - np.min(fi["importance_relative"]))/(np.max(fi["importance_relative"]) - np.min(fi["importance_relative"]))
        fi.to_csv(f"results_all/{dataset_path}/{model_path}/fi_method3{suffix}.csv", index=False)

     

