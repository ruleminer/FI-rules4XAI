import json
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from rmatrix.classification import RMatrixClassifier
from decision_rules.serialization.utils import JSONSerializer

bb_models = pd.read_csv("../results/selected_bb_models.csv")
datasets = bb_models["dataset"].unique()

for sel_dataset in tqdm(datasets, desc="Datasets"):

    for stat in ["precision"]:

        models = np.unique(bb_models[bb_models["dataset"]==sel_dataset]["model"])

        x_train_df = pd.read_csv(f"../results_all/{sel_dataset}/train.csv")
        x_train_df.drop(columns=["target"], inplace=True)

        binary_columns = list(x_train_df.columns[x_train_df.isin([0,1]).all()])
        if len(binary_columns) > 0:
            x_train_df[binary_columns] = x_train_df[binary_columns].astype(str)

        for sel_model in models:

            y_train_df = pd.read_csv(f"../results_all/{sel_dataset}/{sel_model}/preds_train.csv")
            y_train_df = y_train_df.rename(columns={'prediction': 'name'})
            y_train = y_train_df["name"].squeeze().astype(str)

            file_path= f"../results_all/{sel_dataset}/{sel_model}/ruleset_filterFF_{stat}_approx.json"

            if not os.path.exists(file_path):            

                generator = RMatrixClassifier(mincov=3, induction_measuer=stat, filter_duplicates=False, filtration=False, 
                                            cuts_only_between_classes=True, prune=True, max_growing=10)

                model = generator.fit(x_train_df, y_train)
                ruleset = model.ruleset

                ruleset_json = JSONSerializer.serialize(ruleset)
                with open(file_path, 'w') as json_file:
                    json.dump(ruleset_json, json_file)