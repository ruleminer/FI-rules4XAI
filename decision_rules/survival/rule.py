"""
Contains survival rule and conclusion classes.
"""
from typing import List

import numpy as np
from decision_rules import settings
from decision_rules.core.condition import AbstractCondition
from decision_rules.core.rule import AbstractConclusion
from decision_rules.core.rule import AbstractRule
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator


class SurvivalConclusion(AbstractConclusion):
    """Conclusion part of the survival rule

    Args:
        AbstractConclusion (_type_):
    """

    def __init__(
        self,
        estimator: KaplanMeierEstimator,
        column_name: str
    ) -> None:
        super().__init__(estimator, column_name)

    def positives_mask(self, y: np.ndarray) -> np.ndarray:
        # Based on article: Wróbel et al. Learning rule sets from survival data BMC Bioinformatics (2017) 18:285 Page 4 of 13
        # An observation is covered by the rule when it satisfies its premise. The conclusion of r is an estimate Sˆ(T|cj) of the survival function. 
        # Particularly, it is a Kaplan-Meier (KM) estimator [50] calculated on the basis of the instances covered by the rule, that is, satisfying all conditions cj (j = 1, ... , n)

        # Macha: So i think that conclusion positive mask should alwasy be true
        if isinstance(y, np.ndarray):
            return np.ones(y.shape[0], dtype=bool)
        else:
            return True

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AbstractConclusion) and
            other.column_name == self.column_name and
            other.value == self.value
        )

    def __hash__(self) -> int:
        return hash((self.column_name, self.value))

    def __str__(self) -> str:
        return f'{self.column_name} = NaN'


class SurvivalRule(AbstractRule):
    """Survival decision rule.
    """

    def __init__(
        self,
        premise: AbstractCondition,
        conclusion: SurvivalConclusion,
        column_names: List[str]
    ) -> None:
        self.conclusion: SurvivalConclusion = conclusion
        self.measure = None
        super().__init__(premise, conclusion, column_names)


