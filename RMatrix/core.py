from abc import ABC
from abc import abstractmethod

import time
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.classification.rule import ClassificationRule, ClassificationConclusion
from decision_rules.conditions import CompoundCondition, LogicOperators, NominalCondition, ElementaryCondition
from decision_rules.measures import *
from typing import List
import pandas as pd
import numpy as np
import copy
from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import cProfile, pstats
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, cpu_count

class AbstractRMatrix(ABC):

    def __init__(self, mincov: int, induction_measuer: str, filter_duplicates: bool = True, filtration: bool = True, max_growing: int = None, prune: bool = True) -> None:
        self.mincov = mincov
        self.measure_function = globals().get(induction_measuer)
        self._if_filter_duplicates = filter_duplicates    
        self.filtration = filtration
        self.max_growing = max_growing
        self.prune = prune

    @abstractmethod
    def _ruleset_factory(self) -> AbstractRuleSet:
        pass

    def _get_nominal_indexes(self, dataframe: pd.DataFrame) -> list:
        dtype_mask = (dataframe.dtypes == 'object')
        nominal_indexes = np.where(dtype_mask)[0]
        return nominal_indexes.tolist()
    
    def _get_numerical_indexes(self, dataframe: pd.DataFrame) -> list:
        dtype_mask = np.logical_not(dataframe.dtypes == 'object')
        numerical_indexes = np.where(dtype_mask)[0]
        return numerical_indexes.tolist()
    
    def _generate_rules_from_package(self, examples_indexes: int) -> List[AbstractRule]:
        if self.attributes_list is None:
            rules = [self._generate_rule(example_index) for example_index in examples_indexes]
        else:
            if len(self.attributes_list) == 1:
                rules = [self._generate_rule_on_attributes(example_index, [self.columns_names.get_loc(col) for col in self.attributes_list[0]]) for example_index in examples_indexes]
            else:
                rules = [self._generate_rule_on_attributes(example_index, [self.columns_names.get_loc(col) for col in self.attributes_list[example_index]]) for example_index in examples_indexes]
        return rules
    
    @abstractmethod
    def _generate_rule(self, example_index: int) -> AbstractRule:
        pass 
    
    @abstractmethod
    def _grow(self, example_index: int) -> AbstractRule:
        pass 
    
    def _prune(self, rule: AbstractRule):
        
        if len(rule.premise.subconditions) == 1:
            return
        
        continue_pruning = True
        while continue_pruning:
            conditions = rule.premise.subconditions
            quality_best, _ = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
            condition_to_remove = None
            for condition in conditions:
                rule_without_condition = copy.deepcopy(rule)
                rule_without_condition.premise.subconditions.remove(condition)

                quality_without_condition, coverage_without_condition = self._calculate_quality(rule_without_condition, self.X_numpy, self.y_numpy)

                if quality_without_condition >= quality_best:
                    quality_best = quality_without_condition
                    condition_to_remove = condition
                
            if condition_to_remove is None:
                continue_pruning = False 
            else:
                rule.premise.subconditions.remove(condition_to_remove)

            if len(rule.premise.subconditions) == 1:
                continue_pruning = False




    def _induce_condition(self, rule: AbstractRule, example_index: int, X: np.ndarray, y: np.ndarray) -> AbstractCondition:
            quality_best = float("-inf")
            coverage_best = Coverage(0,0,0,0)
            condition_best = None
            examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

            possible_conditions = self._get_possible_conditions(X[example_index], examples_covered_by_rule, y_for_examples_covered_by_rule)
            possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
            if len(possible_conditions_filtered) != 0:
                for condition in possible_conditions_filtered:
                    rule_with_condition = copy.deepcopy(rule)
                    rule_with_condition.premise.subconditions.append(condition)
                    
                    is_example_covered = rule_with_condition.premise.covered_mask(X[example_index].reshape(1,-1))[0] 
                    if is_example_covered:
                        #if (condition.column_index == 0) and (condition.right > 69 and condition.right < 70):
                        #    print("here")
                        quality, coverage = self._calculate_quality(rule_with_condition, X, y)
                        covered_examples = coverage.p + coverage.n
                        if self._check_if_mincov_is_satisfied(covered_examples, y[example_index]):
                            is_decision_attribute_match = rule_with_condition.conclusion.positives_mask(y[example_index])

                            if (quality > quality_best or ((quality == quality_best) and (coverage.p > coverage_best.p))) and is_decision_attribute_match:
                                condition_best = condition
                                quality_best = quality
                                coverage_best = coverage

            return condition_best, quality_best, coverage_best
    
  
    def _check_if_mincov_is_satisfied(self, covered_examples: int, y: str) -> bool:
        return covered_examples >= self.mincov
    
    def _get_covered_examples(self, X: np.ndarray, y: np.ndarray, rule: AbstractRule) -> List[np.ndarray]:
        covered_examples_mask = rule.premise.covered_mask(X)
        return [X[covered_examples_mask], y[covered_examples_mask]]
    
    
    @abstractmethod    
    def _get_possible_conditions(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> List[AbstractCondition]:
        pass

    def _calculate_quality(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray) -> float:
        coverage = rule.calculate_coverage(X=X, y=y)
        quality = self.measure_function(coverage)

        return quality, coverage

    @abstractmethod 
    def _filter_duplicates(self, ruleset: AbstractRuleSet) -> AbstractRuleSet:
        pass

    def _filter_rules(self, ruleset: AbstractRuleSet) -> AbstractRuleSet:
        rules = ruleset.rules 
        rules = sorted(rules, key=lambda rule: (self.measure_function(rule.coverage), coverage(rule.coverage)), reverse=True)
        uncovered_examples = self.X_numpy
        filtered_rules = []
        zero_uncovered = False
        i = 0
        while (not zero_uncovered) and (i < len(rules)): 
            rule = rules[i]
            if len(rule.premise.subconditions)>0:
                mask = rule.premise.covered_mask(uncovered_examples)
                if np.sum(mask) != 0:
                    uncovered_examples = uncovered_examples[~mask]
                    filtered_rules.append(rule)
                    if uncovered_examples.shape[0] == 0:
                        zero_uncovered = True
            i+=1
        return filtered_rules

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:

        self.label = y.name 

        self.X_numpy = X.to_numpy()
        self.y_numpy = y.to_numpy()

        self.attributes_list = attributes_list

        ruleset = self._ruleset_factory()
        
        # get indexes of nominal attributes
        self.nominal_attributes_indexes = self._get_nominal_indexes(X)
        self.numerical_attributes_indexes = self._get_numerical_indexes(X)
        self.columns_names = X.columns

        packages_number = cpu_count()
        package_size = int(np.ceil(self.X_numpy.shape[0] / packages_number))

        indexes = range(0, self.X_numpy.shape[0])
        packages = [indexes[i:i + package_size] for i in range(0, len(indexes), package_size)]

        results = Parallel(n_jobs=-1)(delayed(self._generate_rules_from_package)(package) for package in packages)

        for result in results:
            ruleset.rules.extend(result)
        
        if self._if_filter_duplicates:
            ruleset = self._filter_duplicates(ruleset)

        ruleset.update(X, y, self.measure_function)

        if self.filtration:
            ruleset.rules = self._filter_rules(ruleset)

        self.ruleset = ruleset
        return self

    
    
    
    def local_explanation(self, example_X: pd.Series, example_y: pd.Series, X: pd.DataFrame, y: pd.Series, attributes: list = []) -> AbstractRuleSet:
            
            X = X.append(example_X, ignore_index=True)
            y = y.append(example_y, ignore_index=True)

            self.label = y.name 

            self.X_numpy = X.to_numpy()
            self.y_numpy = y.to_numpy()
        
            self.nominal_attributes_indexes = self._get_nominal_indexes(X)
            self.numerical_attributes_indexes = self._get_numerical_indexes(X)
            attributes_indexes = [X.columns.get_loc(col) for col in attributes]
            self.columns_names = X.columns

            example_index = X.shape[0] - 1

            ruleset = self._ruleset_factory()
            if len(attributes) == 0:
                rule = self._generate_rule(example_index)
            else:
                rule = self._generate_rule_on_attributes(example_index, attributes_indexes)
            #ruleset.rules.append(rule)
            cov = rule.calculate_coverage(X=self.X_numpy, y=self.y_numpy)
            return rule, cov


    @abstractmethod
    def _generate_rule_on_attributes(self, example_index: int, attributes_indexes: list) -> AbstractRule:
        pass 
    

    def _induce_condition_on_attributes(self, rule: AbstractRule, example_index: int, X: np.ndarray, y: np.ndarray, attributes_to_induction) -> AbstractCondition:
        quality_best = float("-inf")
        coverage_best = Coverage(0,0,0,0)
        condition_best = None
        examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

        possible_conditions = self._get_possible_conditions_on_attributes(X[example_index], examples_covered_by_rule, y_for_examples_covered_by_rule, attributes_to_induction)
        possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
        if len(possible_conditions_filtered) != 0:
            for condition in possible_conditions_filtered:
                rule_with_condition = copy.deepcopy(rule)
                rule_with_condition.premise.subconditions.append(condition)
                
                is_example_covered = rule_with_condition.premise.covered_mask(X[example_index].reshape(1,-1))[0] 
                if is_example_covered:
                    #if (condition.column_index == 0) and (condition.right > 69 and condition.right < 70):
                    #    print("here")
                    quality, coverage = self._calculate_quality(rule_with_condition, X, y)
                    covered_examples = coverage.p + coverage.n
                    if self._check_if_mincov_is_satisfied(covered_examples, y[example_index]):
                        is_decision_attribute_match = rule_with_condition.conclusion.positives_mask(y[example_index])

                        if (quality > quality_best or ((quality == quality_best) and (coverage.p > coverage_best.p))) and is_decision_attribute_match:
                            condition_best = condition
                            quality_best = quality
                            coverage_best = coverage

        return condition_best, quality_best, coverage_best    


    @abstractmethod    
    def _get_possible_conditions_on_attributes(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray, attributes_to_induction) -> list:
        pass