"""
Contains condition's classes JSON serializers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from decision_rules.conditions import AbstractCondition
from decision_rules.conditions import AttributesCondition
from decision_rules.conditions import CompoundCondition
from decision_rules.conditions import ElementaryCondition
from decision_rules.conditions import LogicOperators
from decision_rules.conditions import NominalCondition
from decision_rules.serialization.utils import JSONClassSerializer
from decision_rules.serialization.utils import JSONSerializer
from decision_rules.serialization.utils import register_serializer
from pydantic import BaseModel


class _BaseConditionModel(BaseModel):
    type: str
    attributes: List[int]
    negated: bool = False


@dataclass
class _BaseConditionSerializer:  # pylint: disable=too-few-public-methods
    condition_type: str
    condition_class: type


class _NominalConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'elementary_nominal'
    condition_class: type = NominalCondition

    class _Model(_BaseConditionModel):
        value: str

    @staticmethod
    def _from_pydantic_model(model: _NominalConditionSerializer._Model) -> NominalCondition:
        condition = NominalCondition(
            column_index=model.attributes[0],
            value=model.value
        )
        condition.negated = model.negated if model.negated is not None else False
        return condition

    @staticmethod
    def _to_pydantic_model(instance: NominalCondition) -> _NominalConditionSerializer._Model:
        return _NominalConditionSerializer._Model(
            type=_NominalConditionSerializer.condition_type,
            attributes=[instance.column_index],
            negated=instance.negated,
            value=instance.value,
        )


class _ElementaryConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'elementary_numerical'
    condition_class: type = ElementaryCondition

    class _Model(_BaseConditionModel):
        left: Optional[float]
        right: Optional[float]
        left_closed: bool
        right_closed: bool

    @staticmethod
    def _from_pydantic_model(model: _ElementaryConditionSerializer._Model) -> ElementaryCondition:
        left = model.left if model.left is not None else float('-inf')
        right = model.right if model.right is not None else float('inf')
        condition = ElementaryCondition(
            column_index=model.attributes[0],
            left=left,
            right=right,
            left_closed=model.left_closed,
            right_closed=model.right_closed
        )
        condition.negated = model.negated if model.negated is not None else False
        return condition

    @staticmethod
    def _to_pydantic_model(instance: ElementaryCondition) -> _ElementaryConditionSerializer._Model:
        left = instance.left if instance.left != float('-inf') else None
        right = instance.right if instance.right != float('inf') else None
        return _ElementaryConditionSerializer._Model(
            type=_ElementaryConditionSerializer.condition_type,
            attributes=[instance.column_index],
            negated=instance.negated,
            left=left,
            right=right,
            left_closed=instance.left_closed,
            right_closed=instance.right_closed
        )


class _AttributesConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'attributes'
    condition_class: type = AttributesCondition

    class _Model(_BaseConditionModel):
        operator: str

    @staticmethod
    def _from_pydantic_model(model: _AttributesConditionSerializer._Model) -> AttributesCondition:
        condition = AttributesCondition(
            operator=model.operator,
            column_left=model.attributes[0],
            column_right=model.attributes[1]
        )
        condition.negated = model.negated if model.negated is not None else False
        return condition

    @staticmethod
    def _to_pydantic_model(instance: AttributesCondition) -> _AttributesConditionSerializer._Model:
        return _AttributesConditionSerializer._Model(
            type=_AttributesConditionSerializer.condition_type,
            attributes=[instance.column_left, instance.column_right],
            negated=instance.negated,
            operator=instance.operator
        )


class _CompoundConditionSerializer(_BaseConditionSerializer, JSONClassSerializer):

    condition_type: str = 'compound'
    condition_class: type = CompoundCondition

    class _Model(_BaseConditionModel):
        operator: str
        subconditions: List[Any]
        attributes: Optional[List[int]] = []

    @staticmethod
    def _from_pydantic_model(model: _CompoundConditionSerializer._Model) -> CompoundCondition:
        condition = CompoundCondition(
            subconditions=[
                _ConditionSerializer.deserialize(
                    subcondition,
                ) for subcondition in model.subconditions
            ],
            logic_operator=model.operator
        )
        condition.negated = model.negated if model.negated is not None else False
        condition.logic_operator = LogicOperators[condition.logic_operator]
        return condition

    @staticmethod
    def _to_pydantic_model(instance: CompoundCondition) -> _CompoundConditionSerializer._Model:
        subconditions: List[Dict] = []
        attributes = set()
        for subcondition in instance.subconditions:
            subconditions.append(JSONSerializer.serialize(subcondition))
            attributes = attributes.union(subcondition.attributes)
        return _CompoundConditionSerializer._Model(
            type=_CompoundConditionSerializer.condition_type,
            negated=instance.negated,
            operator=instance.logic_operator.value,
            attributes=list(attributes),
            subconditions=subconditions
        )


@register_serializer(NominalCondition)
@register_serializer(ElementaryCondition)
@register_serializer(AttributesCondition)
@register_serializer(CompoundCondition)
class _ConditionSerializer(JSONClassSerializer):

    _elementary_conditions_serializers: List[_BaseConditionSerializer] = [
        _NominalConditionSerializer,
        _ElementaryConditionSerializer,
        _AttributesConditionSerializer,
        _CompoundConditionSerializer
    ]

    _conditions_types_map: Dict[type, JSONClassSerializer] = {
        s.condition_type: s
        for s in _elementary_conditions_serializers
    }
    _conditions_serializers_map: Dict[str, JSONClassSerializer] = {
        s.condition_class: s
        for s in _elementary_conditions_serializers
    }

    @classmethod
    def serialize(cls, instance: AbstractCondition) -> Dict:
        return cls._conditions_serializers_map[instance.__class__].serialize(instance)

    @classmethod
    def deserialize(cls, data: Union[Dict, BaseModel]) -> Any:
        return cls._conditions_types_map[data['type']].deserialize(data)
