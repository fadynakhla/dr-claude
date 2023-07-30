from typing import Dict, List, Union

import loguru
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dr_claude import datamodels
from dr_claude.claude_mcts import action_states


logger = loguru.logger


_decision_template = """You will be given the context of a patient through a list of positive and negative symptoms.
You will then be given a set of symptoms that an intelligent system has predicted are the next best questions to ask the patient.
Your job is to choose the best action.

Known patient state:
positive symptoms: {positive_symptoms}
negative symptoms: {negative_symptoms}

Symptoms to consider ordered by value: {symptoms}

What is the the best symptom to ask the patient about?

Remember to ensure the chosen symptom exactly matches one of those you are asked to consider. Do not provide any other information or text.
Chosen Symptom:
"""

DECISION_PROMPT = PromptTemplate.from_template(_decision_template)


class DecisionClaude:
    def __init__(self):
        self.chain = get_decision_claude()

    def __call__(self, actions: List[Union[datamodels.Symptom, datamodels.Condition]], state):
        valid_actions = [action for action in actions if self.valid_action(action, state)]
        inputs = self.get_action_picker_inputs(valid_actions, state)
        response = self.chain(inputs)
        action = response["text"].strip()
        logger.info(f"Chosen Action: {action}")
        return action

    def get_action_picker_inputs(
        self, actions: List[datamodels.Symptom], state: action_states.SimulationNextActionState
    ) -> Dict[str, str]:
        return {
            "positive_symptoms": " | ".join(
                [action.name for action in state.pertinent_pos]
            ),
            "negative_symptoms": " | ".join(
                [action.name for action in state.pertinent_neg]
            ),
            "symptoms": " | ".join([action.name for action in actions]),
        }

    def valid_action(
        self,
        action: Union[datamodels.Condition, datamodels.Symptom],
        state: action_states.SimulationNextActionState,
    ) -> bool:
        return (
            not isinstance(action, datamodels.Condition)
            and action not in state.pertinent_pos
            and action not in state.pertinent_neg
        )

def get_decision_claude() -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True),
        prompt=DECISION_PROMPT,
        verbose=True,
    )
