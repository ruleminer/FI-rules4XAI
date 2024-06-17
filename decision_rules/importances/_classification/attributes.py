"""
Contains ConditionImportance class for determining importances of condtions in RuleSet.
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
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


class ClassificationRuleSetAttributeImportances(ABC):
    """Classiciation AtrributeImportance allowing to determine importances of atrribute in ClassificationRuleSet
    """

    def calculate_importances_base_on_conditions(self, condition_importances: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate importances of attributes base on condition importances in RuleSet
        """
        attributes_importances = {}

        for class_name, condition_importances_for_class in condition_importances.items():
            attributes_importances_for_class = defaultdict(int)
            for key, value in condition_importances_for_class.items():
                first_element = key.split(' = ')[0]
                attributes_importances_for_class[first_element] += value

            attributes_importances[class_name] = dict(sorted(
                attributes_importances_for_class.items(), key=lambda item: item[1], reverse=True))

        return attributes_importances
