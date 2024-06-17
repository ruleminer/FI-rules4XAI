from abc import ABC
from abc import abstractmethod

import time
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.regression.rule import RegressionRule, RegressionConclusion
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

class RMatrixRegressor(AbstractRMatrix):


      
    def _generate_rule(self, example_index: int) -> RegressionRule:
        rule = RegressionRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=RegressionConclusion(
                value=self.y_numpy[example_index],
                column_name=self.label,
            ))

        rule = self._grow(rule, example_index, self.X_numpy, self.y_numpy)

        if self.prune:
            self._prune(rule)

        return rule
    
    def _grow(self, rule: RegressionRule, example_index: int, X: np.ndarray, y: np.ndarray) -> RegressionRule:

        carry_on = True
        rule_qualities = []

        while(carry_on):

            condition_best, quality_best, coverage_best = self._induce_condition(rule, example_index, X, y)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                self._calculate_quality(rule, X, y)
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
    
    def _ruleset_factory(self) -> RegressionRuleSet:
        return RegressionRuleSet(rules=[], target_column_name=self.label)


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


    def _filter_duplicates(self, ruleset: RegressionRuleSet) -> RegressionRuleSet:
        filtered_ruleset = RegressionRuleSet(rules=[], target_column_name=self.label)
        for rule in ruleset.rules:
            if rule not in filtered_ruleset.rules:
                filtered_ruleset.rules.append(rule)
        return filtered_ruleset
    

    def _generate_rule_on_attributes(self, example_index: int, attributes: list) -> RegressionRule:
        rule = RegressionRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=RegressionConclusion(
                value=self.y_numpy[example_index],
                column_name=self.label,
            ))

        rule = self._grow_on_attributes(rule, example_index, self.X_numpy, self.y_numpy, attributes)
        if self.prune:
            self._prune(rule)
        return rule

    def _grow_on_attributes(self, rule: RegressionRule, example_index: int, X: np.ndarray, y: np.ndarray, attributes: list) -> RegressionRule:

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