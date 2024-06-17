import unittest
import pandas as pd
from scipy.io import arff
import numpy as np
import os 
from pathlib import Path
from rmatrix.survival import RMatrixSurvival
from sklearn.metrics import balanced_accuracy_score
import time
import json

from decision_rules.serialization import JSONSerializer
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TestRMatrixRegressor(unittest.TestCase):

    
    def test_survival_bhs_mincov_5(self):
        mincov = 5
        measure = "log_rank"
        classification_resources = "resources/survival/"
        dataset_path = classification_resources + "BHS_with_group/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)

        ruleset, ibs_train, ibs_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["ibs_train"] == ibs_train
        assert results_saved["ibs_test"] == ibs_test
        assert results_saved["rules_count"] == len(ruleset.rules)
       
        ruleset_dict = dict()
        for rule in ruleset.rules:
            ruleset_dict[str(rule)] = rule.measure

        assert ruleset_saved_json == ruleset_dict


    # def test_ibs_vs_rulekit(self):
    #     mincov = 5
    #     classification_resources = "resources/survival/"
    #     dataset_path = classification_resources + "BHS_with_group/"
    #     dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)

    #     train = pd.read_csv(dataset_path + "train.csv") 
    #     test = pd.read_csv(dataset_path + "test.csv") 

    #     X_train = train.drop(columns=["survival_status"])
    #     y_train = train["survival_status"]

    #     X_test = test.drop(columns=["survival_status"])
    #     y_test = test["survival_status"]

    #     from rulekit import RuleKit
    #     from rulekit.survival import SurvivalRules
    #     from rulekit.params import Measures

    #     RuleKit.init()

    #     model = SurvivalRules(
    #         survival_time_attr = 'survival_time',
    #         min_rule_covered=mincov
    #     )

    #     model.fit(X_train, y_train)

    #     rulekit_ibs_train = model.score(X_train,y_train)
    #     rulekit_ibs_test = model.score(X_test, y_test)

    #     from ruleset_factories.factories.survival import RuleKitRuleSetFactory

    #     factory = RuleKitRuleSetFactory()

    #     decision_rules_ruleset = factory.make(model, X_train, y_train)
    #     ds_rulekit_ibs_train = decision_rules_ruleset.integrated_bier_score(X_train, y_train)
    #     ds_rulekit_ibs_test = decision_rules_ruleset.integrated_bier_score(X_test, y_test)

    #     assert rulekit_ibs_train == ds_rulekit_ibs_train
    #     assert rulekit_ibs_test == ds_rulekit_ibs_test


    def _learn_rmatrix(self, dataset_path, mincov, measure):
        train = pd.read_csv(dataset_path + "train.csv") 
        test = pd.read_csv(dataset_path + "test.csv") 

        X_train = train.drop(columns=["survival_status"])
        y_train = train["survival_status"]

        X_test = test.drop(columns=["survival_status"])
        y_test = test["survival_status"]
  
        generator = RMatrixSurvival(survival_time_attr="survival_time", mincov = mincov, filter_duplicates=True, filtration=True, max_growing=5)
        start_time = time.time()
        model = generator.fit(X_train, y_train)
        end_time = time.time()
        ruleset = model.ruleset
        ibs_train = ruleset.integrated_bier_score(X_train, y_train)
        ibs_test = ruleset.integrated_bier_score(X_test, y_test)

        return ruleset, ibs_train, ibs_test
    


if __name__ == '__main__':
    unittest.main()