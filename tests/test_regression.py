import unittest
import pandas as pd
from scipy.io import arff
import numpy as np
import os 
from pathlib import Path
from rmatrix.regression import RMatrixRegressor
from sklearn.metrics import balanced_accuracy_score
import time
import json
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization import JSONSerializer
from sklearn.metrics import balanced_accuracy_score, r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class TestRMatrixRegressor(unittest.TestCase):

    
    def test_regression_bolts_mincov_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "bolts/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved


    def test_regression_bolts_mincov_5_measure_c2(self):
        mincov = 5
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "bolts/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved

    def test_regression_bolts_mbagrade_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "mbagrade/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved


    def test_regression_mbagrade_mincov_5_measure_c2(self):
        mincov = 5
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "mbagrade/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved

    def test_regression_servo_mbagrade_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "servo/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved


    def test_regression_servo_mincov_5_measure_c2(self):
        mincov = 5
        measure = "c2"
        classification_resources = "resources/regression/"
        dataset_path = classification_resources + "servo/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=RegressionRuleSet)

        ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["r2_score_train"] == r2_score_train
        assert results_saved["r2_score_test"] == r2_score_test
        assert results_saved["mean_absolute_error_train"] == mean_absolute_error_train
        assert results_saved["mean_absolute_error_test"] == mean_absolute_error_test
        assert results_saved["mean_squared_error_train"] == mean_squared_error_train
        assert results_saved["mean_squared_error_test"] == mean_squared_error_test
        assert ruleset == ruleset_saved

    def _learn_rmatrix(self, dataset_path, mincov, measure):
        train = pd.read_csv(dataset_path + "train.csv") 
        test = pd.read_csv(dataset_path + "test.csv") 

        X_train = train.drop(columns=["class"])
        y_train = train["class"]

        X_test = test.drop(columns=["class"])
        y_test = test["class"]
  
        generator = RMatrixRegressor(mincov, measure, filter_duplicates=True, filtration=True)
        start_time = time.time()
        model = generator.fit(X_train, y_train)
        end_time = time.time()
        ruleset = model.ruleset
        r2_score_train = r2_score(y_train, ruleset.predict(X_train))
        r2_score_test = r2_score(y_test, ruleset.predict(X_test))
        mean_absolute_error_train = mean_absolute_error(y_train, ruleset.predict(X_train))
        mean_absolute_error_test = mean_absolute_error(y_test, ruleset.predict(X_test))
        mean_squared_error_train = mean_squared_error(y_train, ruleset.predict(X_train))
        mean_squared_error_test = mean_squared_error(y_test, ruleset.predict(X_test))

        return ruleset, r2_score_train, r2_score_test, mean_absolute_error_train, mean_absolute_error_test, mean_squared_error_train, mean_squared_error_test
    


if __name__ == '__main__':
    unittest.main()