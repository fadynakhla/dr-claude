from typing import Dict, List, Union

import loguru
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dr_claude_old import datamodels
from dr_claude_old.claude_mcts import action_states


logger = loguru.logger


_decision_template = (
    "You are an insightful and inquisitive doctor. "
    "You are with a patient and need to inquire about the patient for enabling the best possible diagnosis."
    "\nYour job is to choose the best symptom to inquire about next."
    "The following symptoms have already been confirmed and rejected, so asking about them again is a waste of time:"
    "\npositive symptoms: {positive_symptoms}"
    "\nnegative symptoms: {negative_symptoms}"
    "You will then be given a set of symptoms that an intelligent system has predicted are the next best questions to ask the patient. "
    "The intelligent system has predicted that the following symptoms are the most valuable to confirm or reject next: {symptoms}. "
    "Your job is to determine which one of the symptoms from the intelligent system that we should inquire about next."
    "\nRemember to ensure that symptom you chose should match exactly one of those that the intelligent system suggested."
    "Now, given the information from above, determine which symptom that is the best to ask from the list predicted by the intelligent system."
    "What is the best symptom to inquire about? Answer by quoting only the name of the symptom."
)
DECISION_PROMPT = PromptTemplate.from_template(_decision_template)


class DecisionClaude:
    def __init__(self):
        self.chain = get_decision_claude()

    def __call__(
        self, actions: List[Union[datamodels.Symptom, datamodels.Condition]], state
    ):
        valid_actions = [
            action for action in actions if self.valid_action(action, state)
        ]
        inputs = self.get_action_picker_inputs(valid_actions, state)
        response = self.chain(inputs)
        action = response["text"].strip()
        logger.info(f"Chosen Action: {action}")
        return action

    def get_action_picker_inputs(
        self,
        actions: List[datamodels.Symptom],
        state: action_states.SimulationNextActionState,
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
