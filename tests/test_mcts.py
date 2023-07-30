import copy
import random
from typing import Callable, Collection, Self

import loguru

from dr_claude import datamodels
from dr_claude import mcts_module
from dr_claude.mcts_module import action_states

import mcts

from dr_claude import kb_reading


class PatientWithConditionMixin(action_states.SimulationMixin):
    """
    A mixin for debugging convergence (patient with a condition)
    """

    def set_condition(
        self, condition: datamodels.Condition, pert_pos: Collection[datamodels.Symptom]
    ) -> None:
        self.condition = condition
        self.true_pertinent_positives = pert_pos

    # def handleSymptom(self, symptom: datamodels.Symptom) -> Self:
    #     next_self = copy.deepcopy(self)
    #     if symptom in next_self.true_pertinent_positives:
    #         next_self.pertinent_pos.add(symptom)
    #     else:
    #         next_self.pertinent_neg.add(symptom)
    #     return next_self


class ConvergenceTestState(
    PatientWithConditionMixin,
    action_states.SimulationNextActionState,
):
    ...


def logtrueconditionhook(
    rollout_policy: action_states.RollOutPolicy, log_freq: int = 10
) -> Callable[[ConvergenceTestState], float]:
    counter = int()

    def log_wrapper(state: ConvergenceTestState) -> float:
        nonlocal counter
        counter += 1
        if not counter % log_freq:
            probas = state.getConditionProbabilityDict()
            true_condition_proba = probas[state.condition]
            loguru.logger.debug(f"{true_condition_proba=} after {counter} rollouts")
        return rollout_policy(state)

    return log_wrapper


def test_convergence():
    ## load the knowledge base
    reader = kb_reading.NYPHKnowldegeBaseReader("data/NYPHKnowldegeBase.html")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)

    ## create the initial state
    conditions = list(matrix.columns.keys())
    the_condition = conditions[0]  ## TODO: INDEX OF CONDITION TO TEST
    the_symptoms = [
        symptom
        for symptom in matrix.rows
        if matrix[symptom, the_condition] > symptom.noise_rate
    ]

    ## Convergence test
    convergence_state = ConvergenceTestState(matrix, discount_rate=1e-9)
    convergence_state.set_condition(the_condition, the_symptoms)
    convergence_state.pertinent_pos.update(random.choices(the_symptoms, k=2))

    ## Rollout policy
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    rollout_policy = logtrueconditionhook(rollout_policy)

    ## create the initial state
    searcher = action_states.MCTS(timeLimit=6000, rolloutPolicy=rollout_policy)

    action = None
    while not isinstance(action, datamodels.Condition):
        action = searcher.search(initialState=convergence_state)
        assert isinstance(action, datamodels.Symptom)
        if action in the_symptoms:
            convergence_state.pertinent_pos.add(action)
        else:
            convergence_state.pertinent_neg.add(action)
        loguru.logger.info(f"action={action}")
    diagnosis = action

    assert (
        diagnosis == the_condition
    ), f"Convergence test failed {diagnosis}!={the_condition}"


test_convergence()
