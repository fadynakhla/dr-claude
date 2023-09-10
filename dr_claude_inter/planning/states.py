"""This module defines the states, actions and MCTS structures."""

from typing import Generic, List, Optional, Set, TypeVar, Union
from typing_extensions import Self
import copy
import abc
import math
import random

from dr_claude_inter import datamodels
from dr_claude_inter.planning import probabilistic

DEFAULT_DISCOUNT_RATE = 0.05

T = TypeVar("T")


class StateBase(abc.ABC, Generic[T]):
    """
    Base class for the state of the game.
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


class DiagnosticStateBase(
    StateBase[Union[datamodels.Symptom, datamodels.Condition]], abc.ABC
):
    """
    Base class for simulating next best action for diagnosis
    """

    diagnosis: Optional[datamodels.Condition] = None

    def takeAction(
        self, action: Union[datamodels.Symptom, datamodels.Condition]
    ) -> "DiagnosticStateBase":
        if isinstance(action, datamodels.Symptom):
            return self.handleSymptom(action)
        elif isinstance(action, datamodels.Condition):
            return self.handleDiagnostic(action)
        else:
            raise ValueError(f"Unknown action type {action}")

    @abc.abstractmethod
    def handleSymptom(self, symptom: datamodels.Symptom) -> "DiagnosticStateBase":
        ...

    @abc.abstractmethod
    def handleDiagnostic(
        self, condition: datamodels.Condition
    ) -> "DiagnosticStateBase":
        ...


"""
    The state of the game is the set of symptoms that have been asked about
"""


class DiagnosticStateWithDynamicModel(DiagnosticStateBase):
    """
    DiagnosticState that uses a dynamics model to simulate
    transitions and rewards.
    """

    def __init__(
        self,
        dynamics: probabilistic.DiagnosticDynamics,
        discount_rate: Optional[float] = None,
        pertinent_positives: Optional[Set[datamodels.Symptom]] = None,
        pertinent_negatives: Optional[Set[datamodels.Symptom]] = None,
    ) -> None:
        self.dynamics = dynamics
        self.pertinent_pos: Set[datamodels.Symptom] = pertinent_positives or set()
        self.pertinent_neg: Set[datamodels.Symptom] = pertinent_negatives or set()
        self.discount_rate = math.exp(-(discount_rate or DEFAULT_DISCOUNT_RATE))
        self.discount_factor = 1

    def increment_discount_factor(self) -> None:
        self.discount_factor *= self.discount_rate

    def getCurrentPlayer(self):
        return 1

    def getPossibleActions(
        self,
    ) -> List[Union[datamodels.Symptom, datamodels.Condition]]:
        return list(
            self.dynamics.symptoms.union(self.dynamics.conditions).difference(
                self.pertinent_neg.union(self.pertinent_pos)
            )
        )

    def handleSymptom(self, symptom: datamodels.Symptom) -> "DiagnosticStateBase":
        next_state = DiagnosticStateWithDynamicModel(
            self.dynamics,
            self.discount_factor,
            copy.copy(self.pertinent_pos),
            copy.copy(self.pertinent_neg),
        )
        proba = self.dynamics.getSymptomProbabilityDict(
            self.pertinent_pos, self.pertinent_neg
        )[symptom]
        if proba > random.uniform(0, 1):
            next_state.pertinent_pos.add(symptom)
        else:
            next_state.pertinent_neg.add(symptom)
        self.increment_discount_factor()
        return next_state

    def handleDiagnostic(
        self, condition: datamodels.Condition
    ) -> "DiagnosticStateBase":
        next_state = DiagnosticStateWithDynamicModel(
            self.dynamics,
            self.discount_factor,
            copy.copy(self.pertinent_pos),
            copy.copy(self.pertinent_neg),
        )
        next_state.diagnosis = condition
        return next_state

    def isTerminal(self) -> bool:
        return self.diagnosis is not None or not len(
            self.dynamics.symptoms.difference(
                self.pertinent_pos.union(self.pertinent_neg)
            )
        )

    def getReward(self) -> float:
        condition_likelihood = self.dynamics.getConditionProbabilityDict(
            self.pertinent_pos, self.pertinent_neg
        )
        assert self.diagnosis is not None
        return condition_likelihood[self.diagnosis]


"""
Rollout policies
"""


class RollOutPolicy(abc.ABC):
    """Interface for an MCTS roll out policy."""

    @abc.abstractmethod
    def __call__(self, state: StateBase) -> float:
        """Given a state, return a reward."""
        ...


class RandomRollOutPolicy(RollOutPolicy):
    """A RollOutPolicy that takes random actions until it
    reaches the terminal state, at which it returns the reward."""

    def __call__(self, state: DiagnosticStateBase) -> float:
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
    """
    A RollOutPolicy that returns the maximum condition likelihood
    in the current state.
    """

    def __call__(self, state: DiagnosticStateWithDynamicModel) -> float:
        condition_likelihoods = state.dynamics.getConditionProbabilityDict(
            state.pertinent_pos, state.pertinent_neg
        )
        return max(condition_likelihoods.values())
