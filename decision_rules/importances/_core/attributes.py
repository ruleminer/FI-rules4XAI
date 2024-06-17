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


class AbstractRuleSetAttributeImportances(ABC):
    """Abstract AtrributeImportance allowing to determine importances of atrribute in RuleSet
    """

    @abstractmethod
    def calculate_importances_base_on_conditions(self, condition_importances: Union[Dict[str, float], Dict[str, Dict[str, float]]]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Calculate importances of attributes base on condition importances in RuleSet
        """
        pass
