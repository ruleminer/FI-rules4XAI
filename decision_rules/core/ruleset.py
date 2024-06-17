"""
Contains abstract ruleset class.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.coverage import CoverageInfoDict
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule
from decision_rules.measures import coverage
from decision_rules.measures import precision


class InvalidStateError(Exception):
    """Error indicating that object in in a wrong state to perform certain
    operation.

    Args:
        Exception (_type_): _description_
    """


class PredictionMapper:
    """Helper class for mapping rule's conclusions to label column values.
    """

    def __init__(self) -> None:
        self._index_to_conclusion_value_map: Dict[int, Any]
        self._conclusion_value_to_index_map: Dict[AbstractConclusion, Any]
        self._unique_conclusions: Set[AbstractConclusion]

    def update_mapping(self, conclusions: List[AbstractConclusion]):
        """Updates rules's conclusions mappings

        Args:
            conclusions (List[AbstractConclusion]): list of rule's conclusions
        """
        self._index_to_conclusion_value_map = {}
        self._conclusion_value_to_index_map = {}
        self._unique_conclusions = set()

        for conclusion in conclusions:
            self._unique_conclusions.add(conclusion)
        for i, conclusion in enumerate(self._unique_conclusions):
            self._index_to_conclusion_value_map[i] = conclusion.value
            self._conclusion_value_to_index_map[conclusion.value] = i
        assert (
            len(self._conclusion_value_to_index_map) ==
            len(self._index_to_conclusion_value_map)
        )

    def get_unique_conclusions(self) -> Iterable[AbstractConclusion]:
        """
        Returns:
            Iterable[AbstractConclusion]: unique conclusions
        """
        return self._unique_conclusions

    def map_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Maps prediction array containing rules's conclusions indices to their values.

        Args:
            prediction (np.ndarray): 1 dimensional array with indices of rule's conclusions.

        Raises:
            InvalidStateError: When no conclusions have been mapped before.

        Returns:
            np.ndarray: Mapped prediction
        """
        if self._index_to_conclusion_value_map is None:
            raise InvalidStateError(
                'You must call "update_mapping" to prepare mapping' +
                'before using this method.'
            )
        return np.vectorize(self._index_to_conclusion_value_map.get)(prediction)

    def map_conclusion_value_to_index(self, conclusion_value: Hashable) -> int:
        """Maps rule conclusion's value to its index.

        Args:
            conclusion_value (Hashable): Rule conclusion's value

        Returns:
            int: Rule conclusion index in mapping
        """
        return self._conclusion_value_to_index_map[conclusion_value]


class AbstractRuleSet(ABC):
    """Abstract ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: List[AbstractRule]
    ) -> None:
        """
        Args:
            rules (List[AbstractRule]):
            dataset_metadata (BaseDatasetMetadata): metadata about datasets
                compatible with model
        """
        self.rules: List[AbstractRule] = rules
        self.column_names: List[str] = None
        self.train_P: Dict[int, int] = None  # pylint: disable=invalid-name
        self.train_N: Dict[int, int] = None  # pylint: disable=invalid-name
        self.default_conclusion: AbstractConclusion
        self._prediction_mapper: PredictionMapper = PredictionMapper()

        self._voting_weights_calculated: bool = False

    @abstractmethod
    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        pass

    def calculate_rules_coverages(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series]
    ):
        """
        Args:
            X_train (Union[np.ndarray, pd.DataFrame]): train dataset
            y_train (Union[np.ndarray, pd.Series]): train labels
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        self._calculate_P_N(*np.unique(y_train, return_counts=True))

        for rule in self.rules:
            P: int = self.train_P[rule.conclusion.value] if self.train_P is not None else None
            N: int = self.train_N[rule.conclusion.value] if self.train_N is not None else None
            rule.coverage = rule.calculate_coverage(
                X_train,
                y_train,
                P=P,
                N=N
            )

    def calculate_rules_weights(
        self,
        measure: Callable[[Coverage], float]
    ):
        """
        Args:
            measure (Callable[[Coverage], float]): quality measure function

        Raises:
            ValueError: if any of the rules in ruleset has uncalculated coverage
        """
        for rule in self.rules:
            if rule.coverage is None:
                raise ValueError(
                    'Tried to calculate voting weight of a rule with uncalculated coverage.' +
                    'You should either call `RuleSet.calculate_rules_coverages` method - to ' +
                    'calculate coverages of all rules - or call `Rule.calculate_coverage` ' +
                    '- to calculate coverage of this specific rule'
                )
            voting_weight = measure(rule.coverage)
            if voting_weight == float('inf'):
                rule.voting_weight = 1
            elif voting_weight == float('-inf'):
                rule.voting_weight = -1 #TODO do weryfikacji dlaczego tak, przy Correlacji (zwracającej -inf) to powoduje dziwne zachowanie przy predykcji, gdy jest pusta reguła
            elif voting_weight == np.nan:
                rule.voting_weight = 0
            else:
                rule.voting_weight = voting_weight
        self._voting_weights_calculated = True

    def _base_update(
        self,
        y_uniques: np.ndarray,
        y_values_count: np.ndarray,
        measure: Callable[[Coverage], float]
    ):
        self._calculate_P_N(y_uniques, y_values_count)
        self._prediction_mapper.update_mapping(
            [rule.conclusion for rule in self.rules] +
            [self.default_conclusion]
        )
        self.calculate_rules_weights(measure)

        self._voting_weights_calculated = True

    def update_using_coverages(
        self,
        coverages_info: Dict[str, CoverageInfoDict],
        columns_names: List[str],
        measure: Callable[[Coverage], float]
    ):
        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )
        if len(self.rules) != len(coverages_info):
            raise ValueError(
                'Length of coverage_info should be the same as number ' +
                f'of rules in ruleset ({len(self.rules)}), is: {len(coverages_info)}'
            )
        self.column_names = columns_names
        y_uniques: List[Any] = []
        y_values_count: List[Any] = []
        for rule in self.rules:
            try:
                coverage_info: CoverageInfoDict = coverages_info[rule.uuid]
                rule.coverage = Coverage(**coverage_info)
                if rule.conclusion.value not in y_uniques:
                    y_uniques.append(rule.conclusion.value)
                    y_values_count.append(rule.coverage.P)
            except KeyError:
                raise ValueError(  # pylint: disable=raise-missing-from
                    f'Coverage info missing for rule: "{rule.uuid}" ' +
                    'and possibly some other rules too.'
                )
        self._base_update(
            np.array(y_uniques),
            np.array(y_values_count),
            measure
        )

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ):
        """Updates ruleset using training dataset. This method should be called
        both after creation of new ruleset or after manipulating any of its rules
        or internal conditions. This method recalculates rules coverages and voting
        weights making it ready for prediction

        Args:
            X_train (pd.DataFrame):
            y_train (pd.Series):
            measure (Callable[[Coverage], float]): voting measure function

        Raises:
            ValueError: if called on empty ruleset with no rules
        """
        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )
        self.column_names = X_train.columns.tolist()
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        y_uniques, y_values_count = np.unique(y_train, return_counts=True)
        self.calculate_rules_coverages(X_train, y_train)

        self._base_update(
            y_uniques,
            y_values_count,
            measure
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Args:
            X (pd.DataFrame)
        Returns:
            np.ndarray: prediction
        """
        # ensure correct column order
        if not self._voting_weights_calculated:
            raise InvalidStateError(
                'Rules coverages must have been calculated before prediction.' +
                'Did you forget to call update(...) method?'
            )
        X = (X[self.column_names]).to_numpy()
        prediction_array = self._prepare_prediction_array(X)
        return self._perform_prediction(X, prediction_array)

    @abstractmethod
    def _perform_prediction(
        self,
        X: pd.DataFrame,
        prediction_array: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _prepare_prediction_array(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[Dict[str, str, float]]:
        """Calculate rules metrics for each rule such as precision,
        coverage, TP, FP etc. This method should be called after updating
        or calculating rules coverages.

        Args:
            metrics_to_calculate (Optional[List[str]], optional): List of metrics names
                to calculate. Defaults to None.

        Raises:
            InvalidStateError: if rule's coverage have not been calculated
        Returns:
            Dict[Dict[str, str, float]]: metrics for each rule
        """

    @abstractmethod
    def calculate_condition_importances(self, X_train: pd.DataFrame, y_train: pd.Series, measure: Callable[[Coverage], float]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Calculate importances of conditions in RuleSet

        Args:
            X_train (pd.DataFrame):
            y_train (pd.Series):
            measure (Callable[[Coverage], float]): measure used to count importance

        Returns:
            Dict[str, float]: condition importances, in the case of classification additionally returns information about class Dict[str, Dict[str, float]]:
        """
        pass

    @abstractmethod
    def calculate_attribute_importances(self, condition_importances: Union[Dict[str, float], Dict[str, Dict[str, float]]]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Calculate importances of attriubtes in RuleSet based on conditions importances

        Args:
            condition_importances Union[Dict[str, float], Dict[str, Dict[str, float]]]: condition importances

        Returns:
            Dict[str, float]: attribute importances, in the case of classification additionally returns information about class Dict[str, Dict[str, float]]:
        """
        pass

    def calculate_ruleset_stats(self) -> Dict[str, float]:
        """Calculate ruleset statistics such as number of rules, average rule length,
        average precision, average coverage. This method should be called after updating
        rules coverages.

        Returns:
            dict: Ruleset statistics
        """
        stats = dict()
        stats["rules_count"] = len(self.rules)
        stats["avg_conditions_count"] = round(
            np.mean([len(rule.premise.subconditions) for rule in self.rules]), 2)
        stats["avg_precision"] = round(
            np.mean([precision(rule.coverage) for rule in self.rules]), 2)
        stats["avg_coverage"] = round(
            np.mean([coverage(rule.coverage) for rule in self.rules]), 2)
        return stats

    def local_explainability(self, X: pd.Series) -> Tuple[List[str], str]:
        """Calculate local explainability of ruleset for given instance.

        Args:
            X (pd.Series): Instance to explain

        Returns:
            list: List of rules covering instance
            str: Decision (in classification task) or prediction (in regression task)
        """
        X = (X[self.column_names]).to_numpy().reshape(1, -1)
        prediction_array = self._prepare_prediction_array(X)
        prediction = self._perform_prediction(X, prediction_array)[0]

        rules_covering_instance = [rule.uuid for rule in self.rules if np.sum(
            rule.premise.covered_mask(X)) == 1]

        return rules_covering_instance, prediction

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractRuleSet) and
            other.rules == self.rules
        )
