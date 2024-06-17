from abc import ABC
from abc import abstractmethod

import time
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
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
import multiprocessing
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')    
from rmatrix.core import AbstractRMatrix
from decision_rules.core.coverage import Coverage

class RMatrixClassifier(AbstractRMatrix):

    def __init__(self, mincov: int, induction_measuer: str, filter_duplicates: bool = True, filtration: bool = True, cuts_only_between_classes: bool = True, max_growing: int = None, prune: bool = True) -> None:
        self.cuts_only_between_classes = cuts_only_between_classes
        super().__init__(mincov, induction_measuer, filter_duplicates, filtration, max_growing, prune)
      
    def _generate_rule(self, example_index: int) -> ClassificationRule:
        rule = ClassificationRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=ClassificationConclusion(
                value=self.y_numpy[example_index],
                column_name=self.label,
            ))

        rule = self._grow(rule, example_index, self.X_numpy, self.y_numpy)
        if self.prune:
            self._prune(rule)

        return rule
    
    def _grow(self, rule: AbstractRule, example_index: int, X: np.ndarray, y: np.ndarray) -> AbstractRule:
        carry_on = True
        rule_qualities = []
        rule_coverages = []
        while(carry_on):
            condition_best, quality_best, coverage_best = self._induce_condition(rule, example_index, X, y)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
                rule_coverages.append(coverage_best)
                if coverage_best.n == 0:
                    carry_on = False
            else:
                carry_on = False

            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False
        
        if len(rule.premise.subconditions) == 0:
            rule.growing_qualities = rule_qualities
            rule.growing_coverages = rule_coverages
            return rule
        else:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]
            rule.growing_qualities = rule_qualities
            rule.growing_coverages = rule_coverages
        return rule
    
    def _check_if_mincov_is_satisfied(self, covered_examples: int, y: str) -> bool:
        return self.decision_attribute_distribution[y] < self.mincov or covered_examples >= self.mincov
    
    def _ruleset_factory(self) -> ClassificationRuleSet:
        unique_values, counts = np.unique(self.y_numpy, return_counts=True)
        self.decision_attribute_distribution = dict(zip(unique_values, counts))
        return ClassificationRuleSet(rules=[])


    def _get_possible_conditions(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> list:
        conditions = []
        conditions.extend([NominalCondition(column_index=indx, value=example[indx]) for indx in self.nominal_attributes_indexes if not pd.isnull(example[indx])])

        for indx in self.numerical_attributes_indexes:
            if self.cuts_only_between_classes:
                attr_values = examples_covered_by_rule[:,indx].astype(float)
                attr_values = np.stack((attr_values, y), axis=1)
                attr_values = attr_values[~pd.isnull(attr_values[:, 0])]
                sorted_indices = np.argsort(attr_values[:, 0])
                sorted_attr_values = attr_values[sorted_indices]
                change_indices = [i for i in range(1, len(sorted_attr_values)) if sorted_attr_values[i, 1] != sorted_attr_values[i-1, 1]]
                mid_points = np.unique([(sorted_attr_values[indx-1,0] + sorted_attr_values[indx,0]) / 2 for indx in change_indices])
            else:
                examples_covered_by_rule_for_attr = examples_covered_by_rule[:,indx].astype(float)
                values = np.sort(np.unique(examples_covered_by_rule_for_attr[~np.isnan(examples_covered_by_rule_for_attr)]))
                mid_points = [(x + y) / 2 for x, y in zip(values, values[1:])]

            conditions.extend([ElementaryCondition(column_index=indx, left_closed=False, right_closed=True, left=float('-inf'), right=mid_point) for mid_point in mid_points])
            conditions.extend([ElementaryCondition(column_index=indx, left_closed=True, right_closed=False, left=mid_point, right=float('inf')) for mid_point in mid_points]) 

        return conditions


    def _filter_duplicates(self, ruleset: ClassificationRuleSet) -> ClassificationRuleSet:
        filtered_ruleset = ClassificationRuleSet(rules=[])
        for rule in ruleset.rules:
            if rule not in filtered_ruleset.rules:
                filtered_ruleset.rules.append(rule)
        return filtered_ruleset
    
  	

  

    def _generate_rule_on_attributes(self, example_index: int, attributes: list) -> ClassificationRule:
        rule = ClassificationRule(
            column_names=self.columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=ClassificationConclusion(
                value=self.y_numpy[example_index],
                column_name=self.label,
            ))

        rule = self._grow_on_attributes(rule, example_index, self.X_numpy, self.y_numpy, attributes)
        if self.prune:
            self._prune(rule)

        return rule
    
    def _grow_on_attributes(self, rule: AbstractRule, example_index: int, X: np.ndarray, y: np.ndarray, attributes: list) -> AbstractRule:
        carry_on = True
        rule_qualities = []
        attributes_to_induction = []
        attributes_to_induction.append(attributes[0])
        i = 0
        while(carry_on):
            condition_best, quality_best, coverage_best = self._induce_condition_on_attributes(rule, example_index, X, y, attributes_to_induction)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
                if coverage_best.n == 0:
                    carry_on = False
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
            rule.growing_qualities = rule_qualities
            return rule
        else:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]
            rule.growing_qualities = rule_qualities
        return rule
    

    
    def _get_possible_conditions_on_attributes(self, example: np.ndarray, examples_covered_by_rule: np.ndarray, y: np.ndarray, attributes_to_induction) -> list:
        nominal_attributes_indexes_to_check = [indx for indx in self.nominal_attributes_indexes if indx in attributes_to_induction]
        numerical_attributes_indexes_to_check = [indx for indx in self.numerical_attributes_indexes if indx in attributes_to_induction]
        conditions = []
        conditions.extend([NominalCondition(column_index=indx, value=example[indx]) for indx in nominal_attributes_indexes_to_check if not pd.isnull(example[indx])])

        for indx in numerical_attributes_indexes_to_check:
            attr_values = examples_covered_by_rule[:,indx].astype(float)
            attr_values = np.stack((attr_values, y), axis=1)
            attr_values = attr_values[~pd.isnull(attr_values[:, 0])]
            sorted_indices = np.argsort(attr_values[:, 0])
            sorted_attr_values = attr_values[sorted_indices]
            change_indices = [i for i in range(1, len(sorted_attr_values)) if sorted_attr_values[i, 1] != sorted_attr_values[i-1, 1]]
            mid_points = np.unique([(sorted_attr_values[indx-1,0] + sorted_attr_values[indx,0]) / 2 for indx in change_indices])
    

            conditions.extend([ElementaryCondition(column_index=indx, left_closed=False, right_closed=True, left=float('-inf'), right=mid_point) for mid_point in mid_points])
            conditions.extend([ElementaryCondition(column_index=indx, left_closed=True, right_closed=False, left=mid_point, right=float('inf')) for mid_point in mid_points]) 

        return conditions