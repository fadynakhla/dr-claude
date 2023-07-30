import copy
from typing import Collection

from dr_claude import datamodels
from dr_claude.mcts import action_states

import mcts

from dr_claude import kb_reading


class PatientWithConditionMixin(action_states.NextBestActionState):
    """
    A mixin for debugging convergence (patient with a condition)
    """

    def set_condition(
        self, condition: datamodels.Condition, pert_pos: Collection[datamodels.Symptom]
    ) -> None:
        self.condition = condition
        self.true_pertinent_positives = pert_pos

    def handleSymptom(
        self, symptom: datamodels.Symptom
    ) -> action_states.NextBestActionState:
        next_self = copy.deepcopy(self)
        if symptom in next_self.true_pertinent_positives:
            next_self.pertinent_pos.add(symptom)
        else:
            next_self.pertinent_neg.add(symptom)
        return next_self

    def getReward(self) -> float:
        return float(self.condition == self.diagnosis)


class ConvergenceTestState(
    PatientWithConditionMixin,
    action_states.SimulationNextActionState,
):
    ...


def test_convergence():
    ## load the knowledge base
    reader = kb_reading.NYPHKnowldegeBaseReader("data/NYPHKnowldegeBase.html")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)

    ## create the initial state
    conditions = list(matrix.columns.keys())
    the_condition = conditions[0]
    the_symptoms = [
        symptom
        for symptom in matrix.rows
        if matrix[symptom, the_condition] > symptom.noise_rate
    ]

    ## create the initial state
    state = ConvergenceTestState(matrix, discount_rate=1e-9)
    state.set_condition(the_condition, the_symptoms)
    searcher = mcts.mcts(timeLimit=10000)
    diagnosis = searcher.search(initialState=state)

    assert (
        diagnosis == the_condition
    ), f"Convergence test failed {diagnosis}!={the_condition}"
