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
from typing import Union

import numpy as np
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractCondition
from decision_rules.core.rule import AbstractRule
from decision_rules.core.ruleset import AbstractRuleSet


@dataclass
class ConditionImportance:
    def __init__(self, condition, quality) -> None:
        self.condition = condition
        self.quality = quality


class AbstractRuleSetConditionImportances(ABC):
    """Abstract ConditionImportance allowing to determine importances of condtions in RuleSet
    """

    def __init__(self, ruleset: AbstractRuleSet):
        """Constructor method
        """
        self.ruleset = ruleset

    @abstractmethod
    def calculate_importances(self, X: np.array, y: np.array, measure: Callable[[Coverage], float]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Calculate importances of conditions in RuleSet
        """
        pass

    def _get_conditions_with_rules(self, rules: List[AbstractRule]) -> Dict[AbstractCondition, List[AbstractRule]]:
        conditions_with_rules = dict()
        for rule in rules:
            rule_conditions = rule.premise.subconditions
            for condition in rule_conditions:
                if condition not in conditions_with_rules.keys():
                    conditions_with_rules[condition] = []
                conditions_with_rules[condition].append(rule)

        return conditions_with_rules

    def _calculate_conditions_importances(self, conditions_with_rules: Dict[str, List[AbstractRule]],  X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> List[ConditionImportance]:
        conditions_importances = []
        for condition in conditions_with_rules.keys():
            sum = 0
            for rule in conditions_with_rules[condition]:
                sum += self._calculate_index_simplified(
                    condition, rule, X, y, measure)
            conditions_importances.append(ConditionImportance(condition, sum))

        return conditions_importances

    @abstractmethod
    def _calculate_index_simplified(self, condition: AbstractCondition, rule: AbstractRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]) -> float:
        pass

    def _calculate_measure(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, measure: Callable[[Coverage], float]):
        return measure(rule.calculate_coverage(X, y))

    @abstractmethod
    def _prepare_importances_dict(self, conditions_importances: Union[List[ConditionImportance], Dict[str, List[ConditionImportance]]]) -> Dict:
        pass
