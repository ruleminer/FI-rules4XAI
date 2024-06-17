import unittest
import pandas as pd
from scipy.io import arff
import numpy as np
import os 
from pathlib import Path
from rmatrix.classification import RMatrixClassifier
from sklearn.metrics import balanced_accuracy_score
import time
import json
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.serialization import JSONSerializer
import warnings
warnings.filterwarnings('ignore')

class TestRMatrixClassifier(unittest.TestCase):

    
    def test_classification_iris_mincov_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "iris/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved

    def test_classification_iris_mincov_5_measure_c2(self):
        mincov = 5
        measure = "c2"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "iris/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved

    def test_classification_iris_mincov_5_measure_correlation(self):
        mincov = 5
        measure = "correlation"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "iris/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved

    def test_classification_iris_mincov_1_measure_correlation(self):
        mincov = 1
        measure = "correlation"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "iris/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved
    
    def test_classification_car_mincov_1_measure_correlation(self):
        mincov = 1
        measure = "correlation"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "car/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved

    def test_classification_car_mincov_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "car/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved
    
    
    def test_classification_autos_mincov_1_measure_c2(self):
        mincov = 1
        measure = "c2"
        classification_resources = "resources/classification/"
        dataset_path = classification_resources + "autos/"
        results_path = dataset_path + f"mincov_{mincov}_measure_{measure}/"
        dataset_path = os.path.join(Path(__file__).resolve().parent, dataset_path)
        results_path = os.path.join(Path(__file__).resolve().parent, results_path)

        with open(results_path+f'/ruleset.json', 'r') as file:
            ruleset_saved_json = json.load(file)
        
        with open(results_path+f'/results.json', 'r') as file:
            results_saved = json.load(file)
       
        ruleset_saved = JSONSerializer.deserialize(ruleset_saved_json, target_class=ClassificationRuleSet)

        ruleset, bacc_train, bacc_test = self._learn_rmatrix(dataset_path, mincov, measure)

        assert results_saved["bacc_train"] == bacc_train
        assert results_saved["bacc_test"] == bacc_test
        assert ruleset == ruleset_saved

    def _learn_rmatrix(self, dataset_path, mincov, measure):
        train = pd.read_csv(dataset_path + "train.csv") 
        test = pd.read_csv(dataset_path + "test.csv") 

        X_train = train.drop(columns=["class"])
        y_train = train["class"].astype(str)

        X_test = test.drop(columns=["class"])
        y_test = test["class"].astype(str)
  
        generator = RMatrixClassifier(mincov, measure, filter_duplicates=True, filtration=True, cuts_only_between_classes=True)
        start_time = time.time()
        model = generator.fit(X_train, y_train)
        end_time = time.time()
        ruleset = model.ruleset
        bacc_train = balanced_accuracy_score(y_train, ruleset.predict(X_train))
        bacc_test = balanced_accuracy_score(y_test, ruleset.predict(X_test))

        return ruleset, bacc_train, bacc_test
    


if __name__ == '__main__':
    unittest.main()