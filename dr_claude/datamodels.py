import collections
from typing import Dict, List, Tuple

import pydantic


class UMLSMixin:
    """
    UMLS Mixin
    """

    umls_code: str

    def __hash__(self) -> int:
        return hash(self.umls_code)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.umls_code == other.umls_code


class Symptom(UMLSMixin, pydantic.BaseModel):
    """
    Symptom
    """

    name: str
    umls_code: str


class Condition(UMLSMixin, pydantic.BaseModel):
    """
    Condition
    """

    name: str
    umls_code: str


class WeightedSymptom(Symptom):
    """
    Weight
    """

    weight: float  # between 0,1


class DiseaseSymptomKnowledgeBase(pydantic.BaseModel):
    pairs: Dict[Condition, List[WeightedSymptom]]
