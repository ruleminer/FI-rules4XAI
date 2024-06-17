"""
Contains regression ruleset class.
"""
from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.coverage import CoverageInfoDict
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule


class RegressionRuleSet(AbstractRuleSet):
    """Regression ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: List[RegressionRule],
        target_column_name: Optional[str] = None
    ) -> None:
        """
        Args:
            rules (List[RegressionRule]):
        """
        self.rules: List[RegressionRule]
        super().__init__(rules)
        if target_column_name is None:
            self.default_conclusion = RegressionConclusion(
                value=0.0,
                column_name=self.rules[0].conclusion.column_name
            )
        else: 
            self.default_conclusion = RegressionConclusion(
                value=0.0,
                column_name=target_column_name
            )

    def update_using_coverages(
        self,
        coverages_info: Dict[str, CoverageInfoDict],
        colum_names: List[str],
        y_median: float,
        measure: Callable[[Coverage], float]
    ):
        self.default_conclusion = RegressionConclusion(
            value=y_median,
            column_name=self.rules[0].conclusion.column_name
        )
        super().update_using_coverages(coverages_info, colum_names, measure)

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ):
        self.default_conclusion = RegressionConclusion(
            value=y_train.median(),
            column_name=self.rules[0].conclusion.column_name
        )
        return super().update(X_train, y_train, measure)

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        return

    def _perform_prediction(
        self,
        X: pd.DataFrame,
        prediction_array: np.ndarray
    ) -> np.ndarray:
        for i in range(len(self.rules)):
            self._predict_with_rule(X, prediction_array, i)
        results_sums: np.ndarray = np.sum(prediction_array[:, :, 0],  axis=1)
        weights_sums: np.ndarray = np.sum(prediction_array[:, :, 1],  axis=1)
        prediction: np.ndarray = np.full(
            shape=(X.shape[0],),
            fill_value=self.default_conclusion.value
        )
        prediction[weights_sums > 0] = results_sums[weights_sums>0] / weights_sums[weights_sums>0]
        return prediction

    def _predict_with_rule(
        self,
        X: np.ndarray,
        prediction_array: np.ndarray,
        rule_index: int
    ):
        rule: RegressionRule = self.rules[rule_index]
        result: float = rule.conclusion.value * rule.voting_weight
        prediction_array[:, rule_index, 0][
            rule.premise.covered_mask(X)
        ] += result
        prediction_array[:, rule_index, 1][
            rule.premise.covered_mask(X)
        ] += rule.voting_weight

    def _prepare_prediction_array(self, X: np.ndarray) -> np.ndarray:
        return np.full(
            (X.shape[0], len(self.rules), 2), fill_value=0.0, dtype=float
        )
    
    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[Dict[str, str, float]]:
        raise NotImplementedError()

    def calculate_condition_importances(self, X_train: pd.DataFrame, y_train: pd.Series, measure: Callable[[Coverage], float]) -> Dict[str, float]:
        raise NotImplementedError("This function is not implemented yet")
    
    def calculate_attribute_importances(self, condition_importances: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError("This function is not implemented yet")