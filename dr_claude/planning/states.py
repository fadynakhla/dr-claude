"""This module defines the states, actions and MCTS structures."""

from typing import Dict, Generic, List, Optional, Set, TypeVar, Union
from typing_extensions import Self
import abc
import copy
import math
import random
import numpy as np

from dr_claude import datamodels
from dr_claude.planning import probabilistic


T = TypeVar("T")


class ActionState(abc.ABC, Generic[T]):
    """
    Base class for the state of the game
    """

    @abc.abstractmethod
    def getCurrentPlayer(self) -> int:
        ...

    @abc.abstractmethod
    def getPossibleActions(self) -> List[T]:
        ...

    @abc.abstractmethod
    def takeAction(self, action: T) -> Self:
        ...

    @abc.abstractmethod
    def isTerminal(self) -> bool:
        ...

    @abc.abstractmethod
    def getReward(self) -> float:
        ...


class NextBestActionState(
    ActionState[Union[datamodels.Symptom, datamodels.Condition]], abc.ABC
):
    """
    base class for simulating next best action for diagnosis
    """

    diagnosis: Optional[datamodels.Condition] = None
    remaining_symptoms: Set[datamodels.Symptom]

    def takeAction(
        self, action: Union[datamodels.Symptom, datamodels.Condition]
    ) -> "NextBestActionState":
        if isinstance(action, datamodels.Symptom):
            return self.handleSymptom(action)
        elif isinstance(action, datamodels.Condition):
            return self.handleDiagnostic(action)
        else:
            raise ValueError(f"Unknown action type {action}")

    @abc.abstractmethod
    def handleSymptom(self, symptom: datamodels.Symptom) -> "NextBestActionState":
        ...

    @abc.abstractmethod
    def handleDiagnostic(
        self, condition: datamodels.Condition
    ) -> "NextBestActionState":
        ...


"""
    The state of the game is the set of symptoms that have been asked about
"""

DEFAULT_DISCOUNT_RATE = 0.1


class SimulationMixin:
    dynamics: datamodels.ProbabilityMatrix
    pertinent_pos: Set[datamodels.Symptom]
    pertinent_neg: Set[datamodels.Symptom]
    remaining_symptoms: Set[datamodels.Symptom]

    def getSymptomProbabilityDict(self) -> Dict[datamodels.Symptom, float]:
        condition_posterior = self.getConditionProbabilityVector()
        symptom_posterior = probabilistic.compute_symptom_posterior_flat_prior_dict(
            matrix=self.dynamics,
            condition_probas=condition_posterior,
        )
        return symptom_posterior

    def getConditionProbabilityVector(self) -> np.ndarray:
        return probabilistic.compute_condition_posterior_flat_prior(
            self.dynamics,
            pertinent_positives=self.pertinent_pos,
            pertinent_negatives=self.pertinent_neg,
        )

    def getConditionProbabilityDict(self) -> Dict[datamodels.Condition, float]:
        return probabilistic.compute_condition_posterior_flat_prior_dict(
            self.dynamics,
            pertinent_positives=self.pertinent_pos,
            pertinent_negatives=self.pertinent_neg,
        )


class SimulationNextActionState(SimulationMixin, NextBestActionState):
    def __init__(
        self,
        matrix: datamodels.ProbabilityMatrix,
        discount_rate: Optional[float] = None,
        pertinent_positives: Optional[Set[datamodels.Symptom]] = None,
        pertinent_negatives: Optional[Set[datamodels.Symptom]] = None,
    ) -> None:
        ## the matrix of probabilities (dynamics function)
        self.dynamics = matrix
        self.conditions = set(matrix.columns.keys())
        self.remaining_symptoms = set(matrix.rows.keys())

        ## consequences of asking a question
        self.pertinent_pos: Set[datamodels.Symptom] = pertinent_positives or set()
        self.pertinent_neg: Set[datamodels.Symptom] = pertinent_negatives or set()

        ## discount factor
        self.discount_rate = math.exp(-(discount_rate or DEFAULT_DISCOUNT_RATE))
        self.discount_factor = 1

    def increment_discount_factor(self) -> None:
        self.discount_factor *= self.discount_rate

    def getCurrentPlayer(self):
        return 1

    def getPossibleActions(
        self,
    ) -> List[Union[datamodels.Symptom, datamodels.Condition]]:
        return list(self.remaining_symptoms.union(self.conditions))

    def handleSymptom(self, symptom: datamodels.Symptom) -> "NextBestActionState":
        next_self = copy.deepcopy(self)
        next_self.remaining_symptoms.remove(symptom)
        proba = self.getSymptomProbabilityDict()[symptom]
        if proba > random.uniform(0, 1):
            next_self.pertinent_pos.add(symptom)
        else:
            next_self.pertinent_neg.add(symptom)
        self.increment_discount_factor()
        return next_self

    def handleDiagnostic(
        self, condition: datamodels.Condition
    ) -> "NextBestActionState":
        next_self = copy.deepcopy(self)
        next_self.diagnosis = condition
        return next_self

    def isTerminal(self) -> bool:
        return len(self.remaining_symptoms) == 0 or self.diagnosis is not None

    def getReward(self) -> float:
        conditions = probabilistic.compute_condition_posterior_flat_prior(
            self.dynamics,
            pertinent_positives=self.pertinent_pos,
            pertinent_negatives=self.pertinent_neg,
        )
        assert self.diagnosis is not None
        return conditions[self.dynamics.columns[self.diagnosis]]


"""
Rollout policy
"""


class RollOutPolicy(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: ActionState) -> float:
        ...


class RandomRollOutPolicy(RollOutPolicy):
    def __call__(self, state: NextBestActionState) -> float:
        while not state.isTerminal():
            try:
                actions_space = state.getPossibleActions()
                action = random.choice(actions_space)
            except IndexError:
                raise Exception(
                    "Non-terminal state has no possible actions: " + str(state)
                )
            state = state.takeAction(action)
        return state.getReward()


class ArgMaxDiagnosisRolloutPolicy(RollOutPolicy):
    def __call__(self, state: SimulationNextActionState) -> float:
        actions_space = state.getConditionProbabilityDict()
        return max(actions_space.values())
