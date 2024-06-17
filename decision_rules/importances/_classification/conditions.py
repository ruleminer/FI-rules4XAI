"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
from decision_rules.classification.rule import ClassificationRule
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.importances._core import AbstractRuleSetConditionImportances
from decision_rules.importances._core import ConditionImportance


class ClassificationRuleSetConditionImportances(AbstractRuleSetConditionImportances):
    """Classification ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def calculate_importances(self, X: np.array, y: np.array, measure: Callable[[Coverage], float]) -> Dict[str, Dict[str, float]]:
        """Calculate importances of conditions in RuleSet
        """
        rules_by_class = self._split_rules_by_decision_class(
            self.ruleset.rules)

        condition_importances_for_classes = {}

        for class_name, class_rules in rules_by_class.items():

            conditions_with_rules = self._get_conditions_with_rules(
                class_rules)
            conditions_importances = self._calculate_conditions_importances(
                conditions_with_rules, X, y, measure
            )
            condition_importances_for_classes[class_name] = conditions_importances

        conditions_importances = self._prepare_importances_dict(
            condition_importances_for_classes
        )

        return conditions_importances

    def _split_rules_by_decision_class(self, rules: List[ClassificationRule]) -> Dict[str, List[ClassificationRule]]:
        rules_by_class = {}
        for rule in rules:
            if rule.conclusion.value not in rules_by_class.keys():
                rules_by_class[rule.conclusion.value] = []
            rules_by_class[rule.conclusion.value].append(rule)
        return rules_by_class

    def _calculate_index_simplified(self, condition: AbstractCondition, rule: ClassificationRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> float:
        rule_conditions = []
        rule_conditions.extend(rule.premise.subconditions)
        number_of_conditions = len(rule_conditions)
        rule_conditions.remove(condition)

        premise_without_evaluated_condition = CompoundCondition(
            subconditions=rule_conditions, logic_operator=rule.premise.logic_operator)

        rule_without_evaluated_condition = ClassificationRule(
            premise_without_evaluated_condition,
            conclusion=rule.conclusion,
            column_names=rule.column_names
        )

        factor = 1.0 / number_of_conditions
        if len(rule_conditions) == 0:
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
            )
        else:
            premise_with_only_evaluated_condition = CompoundCondition(
                subconditions=[condition], logic_operator=rule.premise.logic_operator)

            rule_with_only_evaluated_condition = ClassificationRule(
                premise_with_only_evaluated_condition,
                conclusion=rule.conclusion,
                column_names=rule.column_names
            )
            return factor * (
                self._calculate_measure(rule, X, y, measure)
                - self._calculate_measure(rule_without_evaluated_condition, X, y, measure)
                + self._calculate_measure(rule_with_only_evaluated_condition, X, y, measure)
            )

    def _prepare_importances_dict(self, conditions_importances: Dict[str, List[ConditionImportance]]) -> Dict[str, Dict[str, float]]:

        conditions_importances_sorted = dict()

        for class_name in conditions_importances.keys():

            conditions_importances_for_class_dict = dict()
            condition_importances_for_class_list = conditions_importances[class_name]

            for condition_importance in condition_importances_for_class_list:
                conditions_importances_for_class_dict[condition_importance.condition.to_string(
                    columns_names=self.ruleset.column_names)] = condition_importance.quality

            conditions_importances_sorted[class_name] = dict(
                sorted(
                    conditions_importances_for_class_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        return conditions_importances_sorted
