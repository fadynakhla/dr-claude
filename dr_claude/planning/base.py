"""This module contains the base implementations of the planning module.

The implementation is based on: https://github.com/pbsinclair42/MCTS
but refactored a bit for our purpose.
"""
from typing import Generic, List, Optional, Set, TypeVar, Union
from typing_extensions import Self
from pydantic import BaseModel
import abc

T = TypeVar("T")


class Player(abc.ABC, BaseModel):
    """Representation of a player."""

    id_: int


class MCTS:
    def __init__(self, rollout_policy: "RollOutPolicy") -> None:
        self.rollout_policy = rollout_policy

    def get_next_best_actions(self, inital_state: "MCTSBaseState") -> T:
        ...


class MCTSBaseState(abc.ABC, Generic[T]):
    """Abstract representation of a state used in MCTS."""

    @abc.abstractmethod
    def get_current_player(self) -> Player:
        """Return the player whos turn it is."""
        ...

    @abc.abstractmethod
    def get_possible_actions(self) -> List[T]:
        """Return the action space for the state."""
        ...

    @abc.abstractmethod
    def take_action(self, action: T) -> Self:
        """Take an action and transition to the new state."""
        ...

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the state is terminal (the game is over), else False."""
        ...

    @abc.abstractmethod
    def get_reward(self) -> float:
        """Return the reward scalar for the current state."""
        ...


class RollOutPolicy(abc.ABC):
    """Abstract class for an MCTS roll out policy.

    A policy (distribution of actions given a state) to play until the
    end of the game is reached, in which we return the reward.
    """

    @abc.abstractmethod
    def rollout(self, state: MCTSBaseState) -> float:
        ...
