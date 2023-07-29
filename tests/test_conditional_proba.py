from dr_claude import datamodels
from dr_claude.mcts import base


def test_conditional_condition_proba():
    condition_1 = datamodels.Condition(name="COVID-19", umls_code="C0000001")
    condition_1_differential_symptom = datamodels.WeightedSymptom(
        name="Cough",
        umls_code="C0000001",
        weight=0.5,
    )

    condition_2 = datamodels.Condition(name="Common Cold", umls_code="C0000004")
    condition_2_differential_symptom = datamodels.WeightedSymptom(
        name="Runny nose",
        umls_code="C0000004",
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
    conditional_probas = base.compute_condition_posterior_flat_prior_dict(
        matrix,
        pertinent_positives=[common_symptom, condition_1_differential_symptom],
        pertinent_negatives=[condition_2_differential_symptom],
    )

    assert conditional_probas[condition_1] > conditional_probas[condition_2]


def test_conditional_proba_symptom():
    condition_1 = datamodels.Condition(name="COVID-19", umls_code="C0000001")
    condition_1_differential_symptom = datamodels.WeightedSymptom(
        name="Cough",
        umls_code="C0000001",
        weight=0.5,
    )

    condition_2 = datamodels.Condition(name="Common Cold", umls_code="C0000004")
    condition_2_differential_symptom = datamodels.WeightedSymptom(
        name="Runny nose",
        umls_code="C0000004",
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
    condition_probas = base.compute_condition_posterior_flat_prior(
        matrix,
        pertinent_positives=[common_symptom, condition_1_differential_symptom],
        pertinent_negatives=[condition_2_differential_symptom],
    )
    symptom_probas = base.compute_symptom_posterior_flat_prior_dict(
        matrix, condition_probas
    )

    assert (
        symptom_probas[condition_1_differential_symptom]
        > symptom_probas[condition_2_differential_symptom]
    )
