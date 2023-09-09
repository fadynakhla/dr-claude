from typing import Callable
import random
from loguru import logger

from dr_claude_intermed import datamodels
from dr_claude_intermed.planning import states, probabilistic, drclaude_mcts
from dr_claude_intermed.kb import kb_reading


def logtrueconditionhook(
    true_condition: datamodels.Condition,
    rollout_policy: states.RollOutPolicy,
    log_freq: int = 10,
) -> Callable[[states.DiagnosticStateWithDynamicModel], float]:
    counter = int()

    def log_wrapper(state: states.DiagnosticStateWithDynamicModel) -> float:
        nonlocal counter
        counter += 1
        if not counter % log_freq:
            probas = state.dynamics.getConditionProbabilityDict(
                state.pertinent_pos, state.pertinent_neg
            )
            true_condition_proba = probas[true_condition]
            logger.info(f"{true_condition_proba=} after {counter} rollouts")
        return rollout_policy(state)

    return log_wrapper


def test_convergence():
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    conditions = list(matrix.columns.keys())
    true_condition = conditions[0]  # get a condition
    logger.info("True condition is: {}", true_condition)
    patient_symptoms = [
        symptom
        for symptom in matrix.rows
        if matrix[symptom, true_condition] > symptom.noise_rate
    ]
    dynamics = probabilistic.DiagnosticDynamics(matrix)
    state = states.DiagnosticStateWithDynamicModel(dynamics, discount_rate=1e-9)
    state.pertinent_pos.update(random.choices(patient_symptoms, k=3))

    argmax_rollout_policy = states.ArgMaxDiagnosisRolloutPolicy()
    argmax_rollout_policy = logtrueconditionhook(true_condition, argmax_rollout_policy)
    searcher = drclaude_mcts.DrClaudeMCTS(
        timeLimit=6000, rolloutPolicy=argmax_rollout_policy
    )
    action = None
    while not isinstance(action, datamodels.Condition):
        if action is not None:
            assert isinstance(action, datamodels.Symptom)
        top_actions = searcher.search(initialState=state, top_k=1)
        action = top_actions[0]
        if isinstance(action, datamodels.Condition):
            logger.info("condition prediction: {}", action)
            break
        if action in patient_symptoms:
            logger.info("got a pertinent positive: {}", action)
            state.pertinent_pos.add(action)
        else:
            logger.info("got a pertinent negative: {}", action)
            state.pertinent_neg.add(action)
        logger.info(f"action={action}")
    diagnosis = action

    assert (
        diagnosis == true_condition
    ), f"Convergence test failed {diagnosis}!={true_condition}"
