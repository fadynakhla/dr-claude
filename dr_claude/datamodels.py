import collections
import itertools
from typing import (
    Dict,
    List,
    NamedTuple,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import pydantic


@runtime_checkable
class HasUMLS(Protocol):
    """
    UMLS Mixin
    """

    umls_code: str


HasUMLSClass = TypeVar("HasUMLSClass", bound=HasUMLS)


def set_umls_methods(cls: Type[HasUMLSClass]) -> Type[HasUMLSClass]:
    def __hash__(self: HasUMLSClass) -> int:
        return hash(self.umls_code)

    def __eq__(self: HasUMLSClass, other: object) -> bool:
        return isinstance(other, HasUMLS) and self.umls_code == other.umls_code

    cls.__hash__ = __hash__
    cls.__eq__ = __eq__
    return cls


@set_umls_methods
class Symptom(pydantic.BaseModel):
    """
    Symptom
    """

    name: str
    umls_code: str
    noise_rate: float = 0.03

    class Config:
        frozen = True


@set_umls_methods
class Condition(pydantic.BaseModel):
    """
    Condition
    """

    def __hash__(self) -> int:
        return hash(self.name)

    name: str
    umls_code: str

    class Config:
        frozen = True


@set_umls_methods
class WeightedSymptom(Symptom):
    """
    Weight
    """

    weight: float  # between 0,1

    class Config:
        frozen = True


class DiseaseSymptomKnowledgeBase(pydantic.BaseModel):
    condition_symptoms: Dict[Condition, List[WeightedSymptom]]


"""
Helper methods
"""


class ProbabilityMatrix(NamedTuple):
    """
    A matrix of probabilities
    Rows are symptoms
    Columns are conditions
    """

    matrix: np.ndarray
    rows: Dict[Symptom, int]
    columns: Dict[Condition, int]

    def __getitem__(
        self, item: Tuple[Union[Symptom, slice], Union[Condition, slice]]
    ) -> np.ndarray:
        symptom, condition = item
        if not isinstance(symptom, slice):
            symptom = self.rows[symptom]
        if not isinstance(condition, slice):
            condition = self.columns[condition]
        return self.matrix[symptom, condition]


class MonotonicCounter:
    """
    A counter that increments and returns a new value each time it is called
    """

    def __init__(self, start: int = 0):
        self._count = start

    def __call__(self) -> int:
        c = self._count
        self._count += 1
        return c


class SymptomTransformer:
    @staticmethod
    def to_symptom(symptom: WeightedSymptom) -> Symptom:
        return Symptom(**symptom.dict())


class DiseaseSymptomKnowledgeBaseTransformer:
    @staticmethod
    def to_numpy(kb: DiseaseSymptomKnowledgeBase) -> ProbabilityMatrix:
        """
        Returns a numpy array of the weights of each symptom for each condition
        """

        ## init symptoms
        all_symptoms = itertools.chain.from_iterable(kb.condition_symptoms.values())
        symptom_idx: Dict[Symptom, int] = collections.defaultdict(MonotonicCounter())
        [symptom_idx[s] for s in map(SymptomTransformer.to_symptom, all_symptoms)]

        ## init conditions
        disease_idx: Dict[Condition, int] = collections.defaultdict(MonotonicCounter())
        [disease_idx[condition] for condition in kb.condition_symptoms.keys()]

        ## the antagonist
        probas = np.zeros((len(symptom_idx), len(disease_idx)))

        ## fill noise vals
        for symptom, index in symptom_idx.items():
            probas[index, :] = symptom.noise_rate

        ## fill known probas
        for condition, symptoms in kb.condition_symptoms.items():
            for symptom in symptoms:
                probas[symptom_idx[symptom], disease_idx[condition]] = symptom.weight

        return ProbabilityMatrix(
            matrix=probas,
            rows=dict(symptom_idx),
            columns=dict(disease_idx),
        )
