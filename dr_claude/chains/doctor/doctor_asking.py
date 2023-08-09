"""This module contains the doctor question asking prompts and code."""

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_doc_prompt_template = (
    "You are an insightful and inquisitive doctor. You are with a patient and need to inquire about a specific symptom: {symptom}.\n\n"
    "Compose a single, direct question that exclusively probes the presence of this particular symptom. "
    "Ensure your response contains only this question, with no additional commentary or elements. The entire response should be the question itself."
    "Keep the question simple. For instance, if the symptom was shortness of breath, you can ask 'Are you experiencing any shortness of breath?'"
    "\nIf the symptom was abdominal cramps, you can ask 'Are have you had any abdominal cramps lately?'"
    "\n\nNow, phrase a question that lets you confirm or reject whether the patient has the symptom {symptom}."
    "\n\nQuestion:"
)
DOC_PROMPT = PromptTemplate.from_template(_doc_prompt_template)


def get_doctor_chain(handler: BaseCallbackHandler = BaseCallbackHandler()) -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True, callbacks=[handler]),
        prompt=DOC_PROMPT,
        verbose=True,
        callbacks=[handler],
    )


if __name__ == "__main__":
    doc_chain = get_doctor_chain()
    inputs = {"symptom": "abdominal cramps"}
    print(doc_chain(inputs))
