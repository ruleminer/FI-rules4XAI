"""
Contains classes for regression ruleset JSON serialization.
"""
from __future__ import annotations

from typing import List
from typing import Optional

from decision_rules.regression.rule import RegressionRule
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization._classification.rule import _ClassificationRuleSerializer
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.serialization.utils import register_serializer
from pydantic import BaseModel


class _RegressionMetaDataModel(BaseModel):
    attributes: List[str]
    decision_attribute: str


@register_serializer(RegressionRuleSet)
class _RegressionRuleSetSerializer(JSONClassSerializer):

    class _Model(BaseModel):
        meta: Optional[_RegressionMetaDataModel]
        rules: List[_ClassificationRuleSerializer._Model]

    @classmethod
    def _from_pydantic_model(cls: type, model: _Model) -> RegressionRuleSet:
        ruleset = RegressionRuleSet(
            rules=[
                JSONSerializer.deserialize(
                    rule,
                    RegressionRule
                ) for rule in model.rules
            ],

        )
        ruleset.column_names = model.meta.attributes
        for rule in ruleset.rules:
            rule.column_names = ruleset.column_names
            rule.conclusion.column_name = model.meta.decision_attribute
            rule.train_covered_y_median = rule.conclusion.train_covered_y_median
        return ruleset

    @classmethod
    def _to_pydantic_model(cls: type, instance: RegressionRuleSet) -> _Model:
        if len(instance.rules) == 0:
            raise ValueError('Cannot serialize empty ruleset.')
        return _RegressionRuleSetSerializer._Model(
            meta=_RegressionMetaDataModel(
                attributes=instance.column_names,
                decision_attribute=instance.rules[0].conclusion.column_name
            ),
            rules=[
                JSONSerializer.serialize(rule) for rule in instance.rules
            ]
        )
