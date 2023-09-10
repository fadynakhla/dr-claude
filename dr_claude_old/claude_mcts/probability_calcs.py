from typing import Annotated, Collection, Dict, TypeVar

import numpy as np
from dr_claude_old import datamodels


T = TypeVar("T")


class VectorTransformer:
    @staticmethod
    def to_dictionary(vector: np.ndarray, index: Dict[T, int]) -> Dict[T, float]:
        return {item: vector[i] for item, i in index.items()}


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


if __name__ == "__main__":
    condition_1 = datamodels.Condition(name="COVID-19", umls_code="C0000001")
    condition_1_differential_symptom = datamodels.WeightedSymptom(
        name="Cough",
        umls_code="C0000001",
        weight=0.5,
    )

    condition_2 = datamodels.Condition(name="Common Cold", umls_code="C0000004")
    condition_2_differential_symptom = datamodels.WeightedSymptom(
        name="Runny nose",
        umls_code="C0000003",
        weight=0.5,
    )

    common_symptom = datamodels.WeightedSymptom(
        name="Fever", umls_code="C0000002", weight=0.5
    )

    db = datamodels.DiseaseSymptomKnowledgeBase(
        condition_symptoms={
            condition_1: [common_symptom, condition_1_differential_symptom],
            condition_2: [common_symptom, condition_2_differential_symptom],
        }
    )

    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(db)

    probas = compute_condition_posterior_flat_prior(
        matrix,
        [condition_1_differential_symptom, common_symptom],
        [condition_2_differential_symptom],
    )
    symptom_probas = compute_symptom_posterior_flat_prior(
        matrix,
        probas,
    )
