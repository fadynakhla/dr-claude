from typing import Collection, Dict

import numpy as np
from dr_claude import datamodels


def compute_condition_probabilities(
    matrix: datamodels.ProbabilityMatrix,
    pertinent_positives: Collection[datamodels.Symptom],
    pertinent_negatives: Collection[datamodels.Symptom],
) -> Dict[datamodels.Condition, float]:
    """
    Compute the probability of each condition given the pertinent positives and pertinent negatives
    :param matrix: The probability matrix
    :param pertinent_positives: The pertinent positives
    :param pertinent_negatives: The pertinent negatives
    :return: A dictionary of condition probabilities
    """

    conditional_log_probas = np.zeros(matrix.matrix.shape[1])

    # Compute the log probability of each condition given the pertinent positives and pertinent negatives
    for symptom in pertinent_positives:
        conditional_log_probas += np.log(matrix[symptom, :])
    for symptom in pertinent_negatives:
        conditional_log_probas -= np.log(matrix[symptom, :])

    # Normalize
    conditional_proba_energy = np.exp(conditional_log_probas)
    conditional_probas = conditional_proba_energy / np.sum(conditional_proba_energy)

    return {condition: conditional_probas[i] for condition, i in matrix.columns.items()}


if __name__ == "__main__":
    ...
    db = datamodels.DiseaseSymptomKnowledgeBase(
        condition_symptoms={
            datamodels.Condition(name="COVID-19", umls_code="C0000001"): [
                datamodels.WeightedSymptom(
                    name="Fever",
                    umls_code="C0000002",
                    weight=0.5,
                    noise_rate=0.2,
                ),
                datamodels.WeightedSymptom(
                    name="Cough",
                    umls_code="C0000003",
                    weight=0.5,
                    noise_rate=0.1,
                ),
            ],
            datamodels.Condition(name="Common Cold", umls_code="C0000004"): [
                datamodels.WeightedSymptom(
                    name="Fever",
                    umls_code="C0000002",
                    weight=0.5,
                    noise_rate=0.05,
                ),
                datamodels.WeightedSymptom(
                    name="Runny nose",
                    umls_code="C0000004",
                    weight=0.5,
                    noise_rate=0.01,
                ),
            ],
        }
    )

    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(db)

    conditional_probas = compute_condition_probabilities(
        matrix,
        pertinent_positives=[
            datamodels.Symptom(name="Fever", umls_code="C0000002"),
            datamodels.Symptom(name="Cough", umls_code="C0000003"),
        ],
        pertinent_negatives=[
            datamodels.Symptom(name="Runny nose", umls_code="C0000004"),
        ],
    )

    print(conditional_probas)
