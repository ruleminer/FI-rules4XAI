from abc import ABC
from abc import abstractmethod

import time
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage as CoverageClass
from decision_rules.survival.ruleset import SurvivalRuleSet
from decision_rules.survival.rule import SurvivalRule, SurvivalConclusion
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
import multiprocessing
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')    
from rmatrix.core import AbstractRMatrix
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, cpu_count

class RMatrixSurvival(AbstractRMatrix):

    def __init__(self, survival_time_attr: str, mincov: int, filter_duplicates: bool = True, filtration: bool = True, max_growing: int = None, prune: bool = True) -> None:
        self.survival_time_attr = survival_time_attr
        self.max_growing = max_growing

        super().__init__(mincov, log_rank, filter_duplicates, filtration, max_growing, prune)
        
      
    def _generate_rule(self, example_index: int) -> SurvivalRule:
        km = KaplanMeierEstimator()
        km.fit(self.survival_time, self.survival_status)
        rule = SurvivalRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=SurvivalConclusion(
                estimator=km,
                column_name=self.label,
            ))

        rule = self._grow(rule, example_index, self.X_numpy, self.y_numpy)
        if self.prune:
            self._prune(rule)

        rule = self._update_estimator(rule)

        return rule
    
    def _grow(self, rule: SurvivalRule, example_index: int, X: np.ndarray, y: np.ndarray) -> SurvivalRule:

        carry_on = True
        rule_qualities = []

        while(carry_on):

            condition_best, quality_best, coverage_best = self._induce_condition(rule, example_index, X, y)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
            else:
                carry_on = False

            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False

            
        if len(rule.premise.subconditions) == 0:
            return rule
        else:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]

        return rule
    
    def _update_estimator(self, rule: SurvivalRule) -> SurvivalRule:
        
        covered_examples = rule.premise._calculate_covered_mask(self.X_numpy)
        km = KaplanMeierEstimator()
        km.fit(self.survival_time[covered_examples], self.survival_status[covered_examples])
        rule.conclusion.value = km
        rule.measure, rule.coverage = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
        return rule
    

    def _ruleset_factory(self) -> SurvivalRuleSet:
        return SurvivalRuleSet(rules=[])


    def _get_possible_conditions(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> list:
        conditions = []
        conditions.extend([NominalCondition(column_index=indx, value=example[indx]) for indx in self.nominal_attributes_indexes if not pd.isnull(example[indx])])

        for indx in self.numerical_attributes_indexes:
            examples_covered_by_rule_for_attr = examples_covered_by_rule[:,indx].astype(float)
            values = np.sort(np.unique(examples_covered_by_rule_for_attr[~np.isnan(examples_covered_by_rule_for_attr)]))
            mid_points = [(x + y) / 2 for x, y in zip(values, values[1:])]

            conditions.extend([ElementaryCondition(column_index=indx, left_closed=False, right_closed=True, left=float('-inf'), right=mid_point) for mid_point in mid_points])
            conditions.extend([ElementaryCondition(column_index=indx, left_closed=True, right_closed=False, left=mid_point, right=float('inf')) for mid_point in mid_points]) 

        return conditions


    def _filter_duplicates(self, ruleset: SurvivalRuleSet) -> SurvivalRuleSet:
        filtered_ruleset = SurvivalRuleSet(rules=[])
        for rule in ruleset.rules:
            if rule not in filtered_ruleset.rules:
                filtered_ruleset.rules.append(rule)
        return filtered_ruleset
    
    def _calculate_quality(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray) -> float:

        covered_examples_indexes = np.where(rule.premise._calculate_covered_mask(X))[0]
        uncovered_examples_indexes = np.where(rule.premise._calculate_uncovered_mask(X))[0]
        quality = log_rank(self.survival_time, self.survival_status, covered_examples_indexes, uncovered_examples_indexes)
        coverage = CoverageClass(p=len(covered_examples_indexes), n= 0, P=X.shape[0], N=0)
        return quality, coverage

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:
        ruleset = self._ruleset_factory()
        self.attributes_list = attributes_list
        #self.X_numpy = X.to_numpy()
        self.y_numpy = y.to_numpy()
        self.columns_names = X.columns
        self.survival_time = X[self.survival_time_attr].to_numpy()
        self.survival_status = y.to_numpy()
        #X_without_survival_time = X.drop(self.survival_time_attr, axis=1)
        self.X_numpy =  X.to_numpy()

        self.label = y.name
        # get indexes of nominal attributes
        self.nominal_attributes_indexes = self._get_nominal_indexes(X)
        self.numerical_attributes_indexes = self._get_numerical_indexes(X)
        self.columns_names = X.columns

        survival_time_attr_index = X.columns.get_loc(self.survival_time_attr)
        if survival_time_attr_index in self.nominal_attributes_indexes:
            self.nominal_attributes_indexes.remove(survival_time_attr_index)
        elif survival_time_attr_index in self.numerical_attributes_indexes:
            self.numerical_attributes_indexes.remove(survival_time_attr_index)

        packages_number = cpu_count()
        package_size = int(np.ceil(self.X_numpy.shape[0] / packages_number))

        indexes = range(0, self.X_numpy.shape[0])
        packages = [indexes[i:i + package_size] for i in range(0, len(indexes), package_size)]

        results = Parallel(n_jobs=-1)(delayed(self._generate_rules_from_package)(package) for package in packages)

        for result in results:
            ruleset.rules.extend(result)
        
        ruleset.calculate_rules_coverages(self.X_numpy,self.y_numpy)
        
        if self._if_filter_duplicates:
            ruleset = self._filter_duplicates(ruleset)

        if self.filtration:
            ruleset.rules = self._filter_rules(ruleset)

        ruleset.column_names = self.columns_names 

        ruleset.set_survival_time_attr_name(self.survival_time_attr)
        ruleset.set_survival_status_attr_name(y.name)
        ruleset.update(X,y, self.measure_function)
        self.ruleset = ruleset
        return self
    
    def _filter_rules(self, ruleset: SurvivalRuleSet) -> SurvivalRuleSet:
        rules = ruleset.rules 
        rules = sorted(rules, key=lambda rule: (rule.measure, coverage(rule.coverage)), reverse=True)
        uncovered_examples = self.X_numpy
        filtered_rules = []
        zero_uncovered = False
        i = 0
        while (not zero_uncovered) and (i < len(rules)): 
            rule = rules[i]
            mask = rule.premise.covered_mask(uncovered_examples)
            if np.sum(mask) != 0:
                uncovered_examples = uncovered_examples[~mask]
                filtered_rules.append(rule)
                if uncovered_examples.shape[0] == 0:
                    zero_uncovered = True
            i+=1
        return filtered_rules

    def local_explanation(self, example_X: pd.Series, example_y: pd.Series, X: pd.DataFrame, y: pd.Series, attributes: list = []) -> AbstractRuleSet:
            
        X = X.append(example_X, ignore_index=True)
        y = y.append(example_y, ignore_index=True)

        self.X_numpy = X.to_numpy()
        self.y_numpy = y.to_numpy()
        self.columns_names = X.columns
        self.survival_time = X[self.survival_time_attr].to_numpy()
        self.survival_status = y.to_numpy()

        self.label = y.name
        # get indexes of nominal attributes
        self.nominal_attributes_indexes = self._get_nominal_indexes(X)
        self.numerical_attributes_indexes = self._get_numerical_indexes(X)
        self.columns_names = X.columns

        survival_time_attr_index = X.columns.get_loc(self.survival_time_attr)
        if survival_time_attr_index in self.nominal_attributes_indexes:
            self.nominal_attributes_indexes.remove(survival_time_attr_index)
        elif survival_time_attr_index in self.numerical_attributes_indexes:
            self.numerical_attributes_indexes.remove(survival_time_attr_index)

        attributes_indexes = [X.columns.get_loc(col) for col in attributes]

        example_index = X.shape[0] - 1

        ruleset = self._ruleset_factory()
        if len(attributes) == 0:
            rule = self._generate_rule(example_index)
        else:
            rule = self._generate_rule_on_attributes(example_index, attributes_indexes)
        #ruleset.rules.append(rule)
        cov = rule.calculate_coverage(X=self.X_numpy, y=self.y_numpy)
        return rule, cov
        
    def _generate_rule_on_attributes(self, example_index: int, attributes: list) -> SurvivalRule:
        km = KaplanMeierEstimator()
        km.fit(self.survival_time, self.survival_status)
        rule = SurvivalRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=SurvivalConclusion(
                estimator=km,
                column_name=self.label,
            ))

        rule = self._grow_on_attributes(rule, example_index, self.X_numpy, self.y_numpy, attributes)
        if self.prune:
            self._prune(rule)

        rule = self._update_estimator(rule)

        return rule
    

    def _grow_on_attributes(self, rule: SurvivalRule, example_index: int, X: np.ndarray, y: np.ndarray, attributes: list) -> SurvivalRule:

        carry_on = True
        rule_qualities = []
        attributes_to_induction = []
        attributes_to_induction.append(attributes[0])
        i = 0
        while(carry_on):

            condition_best, quality_best, coverage_best = self._induce_condition_on_attributes(rule, example_index, X, y, attributes_to_induction)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                self._calculate_quality(rule, X, y)
                rule_qualities.append(quality_best)
            else:
                carry_on = False

            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False
            
            attributes_in_rule = [condition.column_index for condition in rule.premise.subconditions]
            if len(attributes)>1:
                if  attributes[i] in attributes_in_rule:
                    if i == len(attributes)-1:
                        carry_on = False
                    else:
                        attributes_to_induction.append(attributes[i+1])
                        i+=1

        if len(rule.premise.subconditions) == 0:
            return rule
        else:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]

        return rule
    
    def _get_possible_conditions_on_attributes(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray, attributes_to_induction) -> list:
        nominal_attributes_indexes_to_check = [indx for indx in self.nominal_attributes_indexes if indx in attributes_to_induction]
        numerical_attributes_indexes_to_check = [indx for indx in self.numerical_attributes_indexes if indx in attributes_to_induction]
        conditions = []
        conditions.extend([NominalCondition(column_index=indx, value=example[indx]) for indx in nominal_attributes_indexes_to_check if not pd.isnull(example[indx])])

        for indx in numerical_attributes_indexes_to_check:
            examples_covered_by_rule_for_attr = examples_covered_by_rule[:,indx].astype(float)
            values = np.sort(np.unique(examples_covered_by_rule_for_attr[~np.isnan(examples_covered_by_rule_for_attr)]))
            mid_points = [(x + y) / 2 for x, y in zip(values, values[1:])]

            conditions.extend([ElementaryCondition(column_index=indx, left_closed=False, right_closed=True, left=float('-inf'), right=mid_point) for mid_point in mid_points])
            conditions.extend([ElementaryCondition(column_index=indx, left_closed=True, right_closed=False, left=mid_point, right=float('inf')) for mid_point in mid_points]) 

        return conditions