"""
Contains classes for regression rule's JSON serialization.
"""
from __future__ import annotations

from typing import Any

from decision_rules.regression.rule import RegressionConclusion
from decision_rules.regression.rule import RegressionRule
from decision_rules.serialization._core.rule import _BaseRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import register_serializer
from pydantic import BaseModel


@register_serializer(RegressionConclusion)
class _RegressionRuleConclusionSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        value: Any
        covered_y_median: float
        covered_y_std: float

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> RegressionConclusion:
        conclusion = RegressionConclusion(
            value=model.value,
            column_name=None
        )
        conclusion.train_covered_y_median = model.covered_y_median
        conclusion.train_covered_y_std = model.covered_y_std
        return conclusion

    @classmethod
    def _to_pydantic_model(
        cls: type,
        instance: RegressionConclusion
    ) -> _Model:
        return _RegressionRuleConclusionSerializer._Model(
            value=instance.value,
            covered_y_median=instance.train_covered_y_median,
            covered_y_std=instance.train_covered_y_std,
            column_name=instance.column_name,
        )


@register_serializer(RegressionRule)
class _RegressionRuleSerializer(_BaseRuleSerializer):
    rule_class = RegressionRule
    conclusion_class = RegressionConclusion
