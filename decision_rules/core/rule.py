"""
Contains abstract rule and conclusion classes.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from uuid import uuid4

import numpy as np
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage


class AbstractConclusion(ABC):
    """Abstract decision rule's conclusion.

    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        value: Any,
        column_name: str
    ) -> None:
        self.value: Any = value
        self.column_name: str = column_name

    @abstractmethod
    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        """Calculates positive examples mask

        Args:
            y (np.ndarray):

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples are consistent with the conclusion.
        """

    def negatives_mask(self, y: np.ndarray) -> np.ndarray:
        """Calculates negatives examples mask - negation of positives mask

        Args:
            y (np.ndarray):

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples are inconsistent with the conclusion.
        """
        return np.logical_not(self.positives_mask(y))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractConclusion) and
            other.column_name == self.column_name and
            other.value == self.value
        )

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class AbstractRule(ABC):
    """Abstract decision rule.

    Args:
        ABC (_type_): _description_
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: AbstractConclusion,
        column_names: List[str]
    ) -> None:
        """
        Args:
            premise (AbstractCondition): condition in premise
            conclusion (int): conclusion decision class
            column_names (List[str]): list of all attributes names in dataset
        """
        self._uuid: str = str(uuid4())
        self.column_names: List[str] = column_names
        self.conclusion: AbstractConclusion = conclusion
        self.premise: AbstractCondition = premise
        self.coverage: Coverage = None
        self.voting_weight: float = 1.0
        self.growing_qualities = []
        self.growing_coverages = []

    @property
    def uuid(self) -> str:
        """Rule uuid

        Returns:
            str: rule uuid
        """
        return self._uuid

    def calculate_coverage(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        P: int = None,
        N: int = None
    ) -> Coverage:
        """
        Args:
            X (np.ndarray)
            y (np.ndarray, optional): if None then `P` and `N` params should
                be passed. Defaults to None.
            P (int, optional): optional number of all examples from rule
                decison class. Defaults to None.
            N (int, optional): optional number of all examples not from rule
                decison class. Defaults to None.

        Raises:
            ValueError: if y is None and either P or N is None too

        Returns:
            Coverage: rule coverage
        """
        if y is None and (P is None or N is None):
            raise ValueError(
                'Either "y" parameter or both "P" and "N" parameters should be passed' +
                'to this method. All of them were None'
            )
        P: int = y[self.conclusion.positives_mask(
            y)].shape[0] if P is None else P
        N: int = y.shape[0] - P if y is not None else N

        with self.premise.cache():
            positive_covered_mask = self.positive_covered_mask(X, y)
            negative_covered_mask = self.negative_covered_mask(X, y)

        p = np.count_nonzero(positive_covered_mask)
        n = np.count_nonzero(negative_covered_mask)

        return Coverage(p, n, P, N)

    def positive_covered_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates positive covered examples mask.

        Args:
            X (np.ndarray)
            y (np.ndarray)

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is positive and covered by a
                rule or not
        """
        return self.conclusion.positives_mask(y) & self.premise.covered_mask(X)

    def negative_covered_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates negative covered examples mask.

        Args:
            X (np.ndarray)
            y (np.ndarray)

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is negative and covered by
                a rule or not
        """
        return self.conclusion.negatives_mask(y) & self.premise.covered_mask(X)

    def positive_uncovered_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates positive uncovered examples mask
        (negation of positive_covered_mask)

        Args:
            X (np.ndarray)
            y (np.ndarray)

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is positive and uncovered by
                a rule or not
        """
        return self.conclusion.positives_mask(y) & self.premise.uncovered_mask(X)

    def negative_uncovered_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates negative uncovered examples mask
        (negation of positive_uncovered_mask)

        Args:
            X (np.ndarray)
            y (np.ndarray)

        Returns:
            np.ndarray: 1 dimensional numpy array of booleans specifying
                whether given examples is negative and uncovered by
                a rule or not
        """
        return self.conclusion.negatives_mask(y) & self.premise.uncovered_mask(X)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractRule) and
            self.conclusion == other.conclusion and
            self.premise == other.premise
        )

    def __str__(self, show_coverage: bool = True) -> str:
        condition_str: str = self.premise.to_string(self.column_names)
        if self.coverage is not None and show_coverage:
            coverage_str: str = str(self.coverage)
        else:
            coverage_str = ''
        return f'IF {condition_str} THEN {str(self.conclusion)} {coverage_str}'
