from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from decision_rules import measures
from decision_rules.classification.rule import ClassificationRule
from decision_rules.conditions import CompoundCondition
from decision_rules.core.coverage import Coverage
from scipy.stats import fisher_exact
from scipy.stats import hypergeom


class RuleMetrics:
    """Class for calculating rule specific metrcis for classification rules.
    All those metrics could be calculated for single rule based on its coverages
    """

    def __init__(self, rule: ClassificationRule) -> None:
        if rule.coverage is None or None in rule.coverage.as_tuple():
            raise ValueError(
                'Cannot calculate rule metrics for rule without calculated coverage.')

        self._metrics_calculator: Dict[str, Callable[[ClassificationRule], Union[float, int]]] = {
            'p': lambda: rule.coverage.p,
            'n': lambda: rule.coverage.n,
            'P': lambda: rule.coverage.P,
            'N': lambda: rule.coverage.N,
            'conditions_count': lambda: self._calculate_conditions_count(rule),
            'precision': lambda: measures.precision(rule.coverage),
            'coverage': lambda: measures.coverage(rule.coverage),
            'C2': lambda: measures.c2(rule.coverage),
            'RSS': lambda: measures.rss(rule.coverage),
            'correlation': lambda: measures.correlation(rule.coverage),
            'lift': lambda: measures.lift(rule.coverage),
            'p_value': lambda: self._fisher_exact_test(rule.coverage),
            'TP': lambda: rule.coverage.p,
            'FP': lambda: rule.coverage.n,
            'TN': lambda: rule.coverage.N - rule.coverage.n,
            'FN': lambda: rule.coverage.P - rule.coverage.p,
            'sensitivity': lambda: measures.sensitivity(rule.coverage),
            'specificity': lambda: measures.specificity(rule.coverage),
            'negative_predictive_value': lambda: self._calculate_negative_predictive_value(rule),
            'odds_ratio': lambda: measures.odds_ratio(rule.coverage),
            'relative_risk': lambda: measures.relative_risk(rule.coverage),
            'lr+': lambda: self._calculate_lr_plus(rule),
            'lr-': lambda: self._calculate_lr_minus(rule),
        }
        self.supported_metrics: List[str] = list(
            self._metrics_calculator.keys())

    def _calculate_lr_plus(self, rule: ClassificationRule) -> float:
        denominator = 1 - measures.specificity(rule.coverage)
        if denominator == 0.0:
            return float('inf')
        return measures.sensitivity(rule.coverage) / denominator

    def _calculate_lr_minus(self, rule: ClassificationRule) -> float:
        denominator = measures.specificity(rule.coverage)
        if denominator == 0.0:
            return float('inf')
        return (1 - measures.sensitivity(rule.coverage)) / denominator

    def _calculate_conditions_count(self, rule: ClassificationRule) -> int:
        if isinstance(rule.premise, CompoundCondition):
            return len(rule.premise.subconditions)
        else:
            return 1

    def _calculate_negative_predictive_value(self, rule: ClassificationRule) -> float:
        """Calculates relative number of correctly as negative classified
        examples among all examples classified as negative

        Args:
            rule (ClassificationRule): rule

        Returns:
            float: negative_predictive_value
        """
        coverage: Coverage = rule.coverage
        tn: int = coverage.N - coverage.n
        fn: int = coverage.P - coverage.p
        return tn / (fn + tn)

    def _fisher_exact_test(self, coverage: Coverage) -> float:
        """Calculates Fisher's exact test for confusion matrix

        Args:
            coverage (Coverage): coverage

        Returns:
            float: p_value
        """
        confusion_matrix = np.array([
            # TP, FP
            [coverage.p, coverage.n],
            # FN, TN
            [coverage.P - coverage.p, coverage.N - coverage.n]]
        )
        M: int = confusion_matrix.sum()
        n: int = confusion_matrix[0].sum()
        N: int = confusion_matrix[:, 0].sum()
        start, end = hypergeom.support(M, n, N)
        hypergeom.pmf(np.arange(start, end+1), M, n, N)
        _, p_value = fisher_exact(confusion_matrix)
        return p_value

    def calculate(
        self,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Union[float, int]]:
        """Calculates metrics for rule.

        Args:
            metrics_to_calculate (Optional[List[str]], optional): List of metrics to calculate.
                Defaults to all possible metrics.

        Raises:
            ValueError: if some unsupported metric name was passed

        Returns:
            Dict[str, Union[float, int]]: dictionary with calculated rule metrics
        """
        if metrics_to_calculate is None:
            metrics_to_calculate = list(self._metrics_calculator.keys())
        try:
            return {
                metric_name: self._metrics_calculator[metric_name]()
                for metric_name in metrics_to_calculate
            }
        except KeyError as error:
            raise ValueError(  # pylint: disable=raise-missing-from
                f'{error} is not a supported metric. ' +
                f'Supported metrics are: {", ".join(self.supported_metrics)}'
            )


class RulesetRulesMetrics:
    """Class for calculating all rules metrics also those for which whole ruleset
    is needed to calculate them. Unline metrics caluclated by `RuleMetrics` class,
    some of the metrics calculated by this class need whole ruleset and dataset to
    be calculated.
    """

    def __init__(
        self,
        rules: List[ClassificationRule],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        if len(rules) == 0:
            raise ValueError('Cannot calculate metrics for empty ruleset.')
        self._X: np.ndarray = X.to_numpy()
        self._y: np.ndarray = y.to_numpy()
        self._rules: List[ClassificationRule] = rules
        self._metrics_calculator: Dict[str, Callable[[dict], Union[float, int]]] = {
            'p_unique': lambda metrics: self._calculate_and_append_uniquely_covered_examples(
                rules,
                self._X,
                self._y,
                metrics,
                covered_type='p'
            ),
            'n_unique': lambda metrics: self._calculate_and_append_uniquely_covered_examples(
                rules,
                self._X,
                self._y,
                metrics,
                covered_type='n'
            ),
        }
        self.supported_metrics: List[str] = (
            RuleMetrics(rule=self._rules[0]).supported_metrics +
            list(self._metrics_calculator.keys())
        )

    def _calculate_and_append_uniquely_covered_examples(
        self,
        rules: List[ClassificationRule],
        X: pd.DataFrame,
        y: pd.Series,
        stats: Dict[str, Dict[str, float]],
        covered_type: str
    ) -> None:
        if covered_type == 'p':
            rules_covered_masks: Dict[str, np.ndarray] = {
                rule.uuid: rule.positive_covered_mask(X, y) for rule in rules
            }
        elif covered_type == 'n':
            rules_covered_masks: Dict[str, np.ndarray] = {
                rule.uuid: rule.negative_covered_mask(X, y) for rule in rules
            }
        else:
            raise ValueError()

        for rule_uuid, rule_p_mask in rules_covered_masks.items():
            others_rules_covered_mask: np.ndarray = np.zeros(
                shape=y.shape[0]).astype(bool)
            for other_rule_uuid, other_rule_covered_mask in rules_covered_masks.items():
                if other_rule_uuid == rule_uuid:
                    continue
                elif others_rules_covered_mask is None:
                    others_rules_covered_mask = other_rule_covered_mask
                else:
                    others_rules_covered_mask |= other_rule_covered_mask
            stats[rule_uuid][f'{covered_type}_unique'] = np.count_nonzero(
                rule_p_mask[np.logical_not(others_rules_covered_mask)]
            )

    def _calculate_rule_specific_metrics(
        self,
        metrics_to_calculate: Optional[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        return {
            rule.uuid: RuleMetrics(rule).calculate(metrics_to_calculate)
            for rule in self._rules
        }

    def calculate(
        self,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculates metrics for rules.

        Args:
            metrics_to_calculate (Optional[List[str]], optional): List of metrics to calculate.
                Defaults to all possible metrics.

        Raises:
            ValueError: if some unsupported metric name was passed

        Returns:
            Dict[str, Union[float, int]]: dictionary with calculated rules metrics, where keys
                are rules uuids and values are dictionaries containing metrics values.
        """
        # divide metrics into rule specific and ruleset specific
        # ruleset specific must be calculated using whole ruleset and dataset
        # rule specific metrics can be calculated only using coverages
        if metrics_to_calculate is None:
            rule_specific_metrics: List[str] = None
            ruleset_specific_metrics: List[str] = None
        else:
            rule_specific_metrics: List[str] = list(
                filter(
                    lambda m: m in RuleMetrics(self._rules[0])
                    .supported_metrics, metrics_to_calculate
                )
            )
            ruleset_specific_metrics: List[str] = list(
                filter(
                    lambda m: m not in RuleMetrics(self._rules[0])
                    .supported_metrics, metrics_to_calculate
                )
            )

        metrics: Dict[str, Dict[str, str, float]] = self._calculate_rule_specific_metrics(
            rule_specific_metrics
        )
        try:
            if ruleset_specific_metrics is None:
                ruleset_specific_metrics = list(
                    self._metrics_calculator.keys())
            for ruleset_specific_metric in ruleset_specific_metrics:
                self._metrics_calculator[ruleset_specific_metric](metrics)
        except KeyError as error:
            raise ValueError(  # pylint: disable=raise-missing-from
                f'{error} is not a supported metric. ' +
                f'Supported metrics are: {", ".join(self.supported_metrics)}'
            )
        return metrics
