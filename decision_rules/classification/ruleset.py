"""
Contains classification ruleset class.
"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.classification.metrics import RulesetRulesMetrics
from decision_rules.classification.rule import ClassificationConclusion
from decision_rules.classification.rule import ClassificationRule
from decision_rules.core.coverage import Coverage
from decision_rules.core.coverage import CoverageInfoDict
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.importances._classification.attributes import ClassificationRuleSetAttributeImportances
from decision_rules.importances._classification.conditions import ClassificationRuleSetConditionImportances


class ClassificationRuleSet(AbstractRuleSet):
    """Classification ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: List[ClassificationRule],
    ) -> None:
        """
        Args:
            rules (List[ClassificationRule]):
        """
        self.rules: List[ClassificationRule]
        self.y_values: np.ndarray = None
        super().__init__(rules)

    def update_using_coverages(
        self,
        coverages_info: Dict[str, CoverageInfoDict],
        columns_names: List[str],
        measure: Callable[[Coverage], float]
    ):
        super().update_using_coverages(coverages_info, columns_names, measure)
        self.y_values = np.unique(
            list(map(lambda c: c.value, self._prediction_mapper.get_unique_conclusions()))
        )

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ):
        super().update(X_train, y_train, measure)
        self.y_values = np.unique(y_train)

    def _update_majority_class(self):
        majority_class: Any = max(self.train_P, key=self.train_P.get)
        self.default_conclusion = ClassificationConclusion(
            value=majority_class,
            column_name=self.rules[0].conclusion.column_name
        )

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        all_counts: int = np.sum(y_values_count)
        self.train_P = {}
        self.train_N = {}
        for i, value in enumerate(y_uniques):
            self.train_P[value] = y_values_count[i]
            self.train_N[value] = all_counts - y_values_count[i]
        self._update_majority_class()

    def _perform_prediction(
        self,
        X: pd.DataFrame,
        prediction_array: np.ndarray
    ) -> np.ndarray:
        for i in range(len(self.rules)):
            self._predict_with_rule(X, prediction_array, i)
        not_covered_examples_mask = np.all(
            np.isclose(prediction_array, 0.0), axis=1)
        prediction = np.argmax(prediction_array, axis=1)
        # map prediction values indices for real values
        prediction = self._prediction_mapper.map_prediction(prediction)
        # predict uncovered examples with default conclusion
        prediction[not_covered_examples_mask] = self.default_conclusion.value
        return prediction

    def _predict_with_rule(
        self,
        X: np.ndarray,
        prediction_array: np.ndarray,
        rule_index: int
    ):
        rule: ClassificationRule = self.rules[rule_index]
        conclusion_index: int = self._prediction_mapper\
            .map_conclusion_value_to_index(rule.conclusion.value)
        prediction_array[:, conclusion_index][
            rule.premise.covered_mask(X)
        ] += rule.voting_weight

    def _prepare_prediction_array(self, X: np.ndarray) -> np.ndarray:
        prediction_array_shape = (
            X.shape[0],
            len(self._prediction_mapper.get_unique_conclusions())
        )
        return np.full(
            prediction_array_shape, fill_value=0.0, dtype=float
        )

    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[Dict[str, str, float]]:
        return RulesetRulesMetrics(self.rules, X, y).calculate(
            metrics_to_calculate
        )

    def calculate_condition_importances(self, X_train: pd.DataFrame, y_train: pd.Series, measure: Callable[[Coverage], float]) -> Dict[str, Dict[str, float]]:
        condtion_importances_generator = ClassificationRuleSetConditionImportances(
            self)
        self.condition_importances = condtion_importances_generator.calculate_importances(
            X_train.to_numpy(), y_train.to_numpy(), measure)
        return self.condition_importances

    def calculate_attribute_importances(self, condition_importances: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        attributes_importances_generator = ClassificationRuleSetAttributeImportances()
        self.attribute_importances = attributes_importances_generator.calculate_importances_base_on_conditions(
            condition_importances)
        return self.attribute_importances
