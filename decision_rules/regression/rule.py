"""
Contains regression rule and conclusion classes.
"""
from typing import List
import numpy as np
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.coverage import Coverage
from decision_rules.core.rule import AbstractRule, AbstractConclusion
from decision_rules import settings


class RegressionConclusion(AbstractConclusion):
    """Conclusion part of the regression rule

    Args:
        AbstractConclusion (_type_):
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        value: float,
        column_name: str
    ) -> None:
        self._value: float = value
        self._low: float = None
        self._high: float = None
        self.column_name: str = column_name
        self._train_covered_y_std: float = None
        self.train_covered_y_median: float = None

    @property
    def value(self) -> float:
        """
        Returns:
            float: Conclusion's value
        """
        return self._value

    @value.setter
    def value(self, new_value: float):
        self._value = new_value
        if self._train_covered_y_std is not None:
            self._calculate_low_high()

    @property
    def low(self) -> float:
        """
        Returns:
            float: Conclusion's lower boundary value
        """
        return self._low

    @property
    def high(self) -> float:
        """
        Returns:
            float: Conclusion's higher boundary value
        """
        return self._high

    @property
    def train_covered_y_std(self) -> float:
        """
        Returns:
            float: Standard deviation of covered examples labels from 
                training dataset.
        """
        return self._train_covered_y_std

    @train_covered_y_std.setter
    def train_covered_y_std(self, value: float) -> float:
        self._train_covered_y_std = value
        self._calculate_low_high()

    def _calculate_low_high(self):
        self._low = self._value - self._train_covered_y_std
        self._high = self._value + self._train_covered_y_std

    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        return np.abs(y - self.train_covered_y_median) <= self._train_covered_y_std

    def __hash__(self) -> int:
        return hash((self._value, self.column_name))

    def __str__(self) -> str:
        return (
            f'{self.column_name} = {{{self._value:,.{settings.FLOAT_DISPLAY_PRECISION}}}} ' +
            f'[{self._low:,.{settings.FLOAT_DISPLAY_PRECISION}}, ' +
            f'{self._high:,.{settings.FLOAT_DISPLAY_PRECISION}}]'
        )


class RegressionRule(AbstractRule):
    """Regression rule.
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: RegressionConclusion,
        column_names: List[str]
    ) -> None:
        self.conclusion: RegressionConclusion = conclusion
        super().__init__(premise, conclusion, column_names)

        self.train_covered_y_median: float = None

    def calculate_coverage(
            self,
            X: np.ndarray,
            y: np.ndarray = None,
            P: int = None,
            N: int = None
    ) -> Coverage:
        covered_y: np.ndarray = y[self.premise.covered_mask(X)]
        self.conclusion.train_covered_y_median = np.median(covered_y)
        self.conclusion._value = self.conclusion.train_covered_y_median 
        y_mean: float = np.mean(covered_y)
        self.conclusion.train_covered_y_std = np.sqrt(
            (np.sum(np.square(covered_y)) /
             covered_y.shape[0]) - (y_mean * y_mean)
        )
        return super().calculate_coverage(X, y, P, N)
