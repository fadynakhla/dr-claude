from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_action_explainer_template = """You will be given the context of a patient through a list of positive and negative symptoms.
You will then be given a symptom that an intelligent system has predicted is the next best question to ask the patient.
Your job is to explain why this is the best question to ask the patient.

Patient state:
positive symptoms: {positive_symptoms}
negative symptoms: {negative_symptoms}

Next best question: {symptom}

Why is this the best question to ask the patient?
"""

ACTION_EXPLAINER_PROMPT = PromptTemplate.from_template(_action_explainer_template)


def get_explain_yourself() -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True),
        prompt=ACTION_EXPLAINER_PROMPT,
        verbose=True,
    )
