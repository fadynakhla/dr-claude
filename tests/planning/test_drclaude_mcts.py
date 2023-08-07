from typing import Callable, Collection
import random
import loguru
from loguru import logger

from dr_claude import datamodels
from dr_claude.planning import states, probabilistic
from dr_claude.kb import kb_reading


class PatientWithConditionMixin:
    """
    A mixin for debugging convergence (patient with a condition)
    """

    def set_condition(
        self, condition: datamodels.Condition, pert_pos: Collection[datamodels.Symptom]
    ) -> None:
        self.condition = condition
        self.true_pertinent_positives = pert_pos


class ConvergenceTestState(
    PatientWithConditionMixin,
    states.DiagnosticStateWithDynamicModel,
):
    ...


def logtrueconditionhook(
    rollout_policy: states.RollOutPolicy, log_freq: int = 10
) -> Callable[[ConvergenceTestState], float]:
    counter = int()

    def log_wrapper(state: ConvergenceTestState) -> float:
        nonlocal counter
        counter += 1
        if not counter % log_freq:
            probas = state.dynamics.getConditionProbabilityDict(
                state.pertinent_pos, state.pertinent_neg
            )
            true_condition_proba = probas[state.condition]
            loguru.logger.debug(f"{true_condition_proba=} after {counter} rollouts")
        return rollout_policy(state)

    return log_wrapper


def test_convergence():
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    conditions = list(matrix.columns.keys())
    the_condition = conditions[0]  # get the first condition
    the_symptoms = [
        symptom
        for symptom in matrix.rows
        if matrix[symptom, the_condition] > symptom.noise_rate
    ]
    dynamics = probabilistic.DiagnosticDynamics(matrix)
    state = ConvergenceTestState(dynamics, discount_rate=1e-9)

    state.set_condition(the_condition, the_symptoms)
    state.pertinent_pos.update(random.choices(the_symptoms, k=1))

    ## Rollout policy
    rollout_policy = states.ArgMaxDiagnosisRolloutPolicy()
    rollout_policy = logtrueconditionhook(rollout_policy)

    ## create the initial state
    searcher = mcts.mcts(timeLimit=3000, rolloutPolicy=rollout_policy)

    action = None
    while not isinstance(action, datamodels.Condition):
        if action is not None:
            assert isinstance(action, datamodels.Symptom)
        action = searcher.search(initialState=state)
        if action in the_symptoms:
            logger.info("got a pertinent positive: {}", action)
            state.pertinent_pos.add(action)
        else:
            logger.info("got a pertinent negative: {}", action)
            state.pertinent_neg.add(action)
        loguru.logger.info(f"action={action}")
    diagnosis = action

    assert (
        diagnosis == the_condition
    ), f"Convergence test failed {diagnosis}!={the_condition}"


test_convergence()
