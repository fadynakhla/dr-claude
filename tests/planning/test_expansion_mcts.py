from typing import Callable, Collection, Dict
import random

import loguru
from loguru import logger

from dr_claude_old import datamodels
from dr_claude_old.claude_mcts import action_states

import mcts

from dr_claude_old import kb_reading


class SymptomLikelihoodPriorExpansionMixin:
    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)

    def expand(self, node):
        """
        The default method will simply expand one action at a time and return them. However,
        to make this more efficient, we should expand all the actions at once and set a good prior
        for them, then use UCT to choose the best child.
        """
        actions = node.state.getPossibleActions()
        symptom_probs: Dict[
            datamodels.Symptom, float
        ] = node.state.getSymptomProbabilityDict()
        condition_probs: Dict[
            datamodels.Condition, float
        ] = node.state.getConditionProbabilityDict()
        for action in actions:
            if action not in node.children:
                child_node = mcts.treeNode(node.state.takeAction(action), node)
                child_node.numVisits = 1
                if child_node.isTerminal:
                    child_node.totalReward += condition_probs[action] * 0.65
                else:
                    child_node.totalReward += symptom_probs[action]
                node.children[action] = child_node
        if len(actions) == len(node.children):
            node.isFullyExpanded = True
        if node.numVisits == 0:
            node.numVisits += len(node.children)
        return self.getBestChild(node, self.explorationConstant)


class PatientWithConditionMixin(action_states.SimulationMixin):
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
    action_states.SimulationNextActionState,
):
    ...


def logtrueconditionhook(
    rollout_policy: action_states.RollOutPolicy, log_freq: int = 1
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


class ImprovedExpansionMCTS(SymptomLikelihoodPriorExpansionMixin, mcts.mcts):
    ...


def test_convergence():
    ## load the knowledge base
    # reader = kb_reading.NYPHKnowldegeBaseReader("data/NYPHKnowldegeBase.html")
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    ## create the initial state
    conditions = list(matrix.columns.keys())
    the_condition = conditions[20]  # TODO chose condiiton here
    print("the codition", the_condition)
    import random

    the_symptoms = [
        symptom
        for symptom in matrix.rows
        if matrix[symptom, the_condition] > random.uniform(0, 1)
    ]
    print("The symptoms:", the_symptoms)
    state = ConvergenceTestState(matrix, discount_rate=1e-9)
    # state = action_states.SimulationNextActionState(matrix, discount_rate=1e-9)

    state.set_condition(the_condition, the_symptoms)
    import random

    random.seed(12)
    state.pertinent_pos.update(random.choices(the_symptoms, k=1))
    print("initial state:", state.pertinent_pos)

    ## Rollout policy
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    rollout_policy = logtrueconditionhook(rollout_policy)

    ## create the initial state
    searcher = ImprovedExpansionMCTS(timeLimit=3000, rolloutPolicy=rollout_policy)

    action = None
    while not isinstance(action, datamodels.Condition):
        if action is not None:
            assert isinstance(action, datamodels.Symptom)
        action = searcher.search(initialState=state)
        if isinstance(action, datamodels.Condition):
            logger.info("predicting diease: {}", action)
            return
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
