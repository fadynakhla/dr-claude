from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_decision_template = """You will be given the context of a patient through a list of positive and negative symptoms.
You will then be given a set of symptoms that an intelligent system has predicted are the next best questions to ask the patient.
Your job is to choose the best action.

Known patient state:
positive symptoms: {positive_symptoms}
negative symptoms: {negative_symptoms}

Symptoms to consider: {symptoms}

What is the the best symptom to ask the patient about?
Remember to output only one symptom and ensure it exactly matches one of those provided.

Answer:
"""

DECISION_PROMPT = PromptTemplate.from_template(_decision_template)


def get_decision_claude() -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True),
        prompt=DECISION_PROMPT,
        verbose=True,
    )
