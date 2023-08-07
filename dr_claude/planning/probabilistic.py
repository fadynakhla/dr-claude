"""This module contains functions and classes to deal with probabilistic calculations."""

from typing import Annotated, Collection, Dict, TypeVar, Set
import numpy as np

from dr_claude import datamodels


T = TypeVar("T")


class VectorTransformer:
    @staticmethod
    def to_dictionary(vector: np.ndarray, index: Dict[T, int]) -> Dict[T, float]:
        return {item: vector[i] for item, i in index.items()}


class DiagnosticDynamics:
    """A class that exposes useful dynamics functions over a probability matrix."""

    def __init__(self, probability_matrix: datamodels.ProbabilityMatrix) -> None:
        self._probability_matrix = probability_matrix

    @property
    def conditions(self) -> Set[datamodels.Condition]:
        return set(self._probability_matrix.columns.keys())

    @property
    def symptoms(self) -> Set[datamodels.Symptom]:
        return set(self._probability_matrix.rows.keys())

    def getSymptomProbabilityDict(
        self,
        pertinent_positives: Collection[datamodels.Symptom],
        pertinent_negatives: Collection[datamodels.Symptom],
    ) -> Dict[datamodels.Symptom, float]:
        condition_posterior = self.getConditionProbabilityVector(
            pertinent_positives, pertinent_negatives
        )
        symptom_posterior = compute_symptom_posterior_flat_prior_dict(
            matrix=self._probability_matrix,
            condition_probas=condition_posterior,
        )
        return symptom_posterior

    def getConditionProbabilityVector(
        self,
        pertinent_positives: Collection[datamodels.Symptom],
        pertinent_negatives: Collection[datamodels.Symptom],
    ) -> np.ndarray:
        return compute_condition_posterior_flat_prior(
            self._probability_matrix,
            pertinent_positives=pertinent_positives,
            pertinent_negatives=pertinent_negatives,
        )

    def getConditionProbabilityDict(
        self,
        pertinent_positives: Collection[datamodels.Symptom],
        pertinent_negatives: Collection[datamodels.Symptom],
    ) -> Dict[datamodels.Condition, float]:
        return compute_condition_posterior_flat_prior_dict(
            self._probability_matrix,
            pertinent_positives=pertinent_positives,
            pertinent_negatives=pertinent_negatives,
        )


def compute_symptom_posterior_flat_prior_dict(
    matrix: datamodels.ProbabilityMatrix,
    condition_probas: Annotated[np.ndarray, "p(condition)"],
) -> Annotated[
    Dict[datamodels.Symptom, float],
    "p(symptom) | (pertinent_positives, pertinent_negatives)",
]:
    """
    Compute the probability of each symptom given a set of condition probabiltiiies
    returns a vector of size m with p(symptom) | conditional_probas where m is the number of conditions
    ASSUMES THAT ALL CONDITIONS ARE EQUALLY LIKELY (p(c) = 1 / n)
    """

    symptom_probas = compute_symptom_posterior_flat_prior(matrix, condition_probas)
    return VectorTransformer.to_dictionary(symptom_probas, matrix.rows)


def compute_symptom_posterior_flat_prior(
    matrix: datamodels.ProbabilityMatrix,
    condition_probas: Annotated[np.ndarray, "p(condition)"],
) -> Annotated[np.ndarray, "p(symptom) | (pertinent_positives, pertinent_negatives)"]:
    """
    Compute the probability of each symptom given a set of condition probabiltiiies
    returns a vector of size m with p(symptom) | conditional_probas where m is the number of conditions
    ASSUMES THAT ALL CONDITIONS ARE EQUALLY LIKELY (p(c) = 1 / n)
    """
    return np.sum(matrix.matrix * condition_probas, axis=1)


def compute_condition_posterior_flat_prior_dict(
    matrix: datamodels.ProbabilityMatrix,
    pertinent_positives: Collection[datamodels.Symptom],
    pertinent_negatives: Collection[datamodels.Symptom],
) -> Annotated[
    Dict[datamodels.Condition, float],
    "p(condition) | (pertinent_positives, pertinent_negatives)",
]:
    condition_probas = compute_condition_posterior_flat_prior(
        matrix,
        pertinent_positives,
        pertinent_negatives,
    )
    return VectorTransformer.to_dictionary(condition_probas, matrix.columns)


def compute_condition_posterior_flat_prior(
    matrix: datamodels.ProbabilityMatrix,
    pertinent_positives: Collection[datamodels.Symptom],
    pertinent_negatives: Collection[datamodels.Symptom],
) -> Annotated[np.ndarray, "p(condition) | (pertinent_positives, pertinent_negatives)"]:
    """
    Compute the probability of each condition given the pertinent positives and pertinent negatives
    returns a vector of size n with p(condition) | (pertinent_positives, pertinent_negatives) where n is the number of conditions
    ASSUMES THAT ALL CONDITIONS ARE EQUALLY LIKELY (p(c) = 1 / n)
    """

    # Compute the log probability of each condition given the pertinent positives and pertinent negatives
    conditional_log_probas = np.zeros(matrix.matrix.shape[1])
    for symptom in pertinent_positives:
        conditional_log_probas += np.log(matrix[symptom, :])
    for symptom in pertinent_negatives:
        conditional_log_probas += np.log(1 - matrix[symptom, :])

    # Normalize
    conditional_proba = np.exp(conditional_log_probas)
    conditional_proba /= np.sum(conditional_proba)
    return conditional_proba
