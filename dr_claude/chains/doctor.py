from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_doc_prompt_template = (
    "You are an insightful and inquisitive doctor. You are with a patient and need to inquire about a specific symptom: {symptom}.\n\n"
    "Compose a single, direct question that exclusively probes the presence of this particular symptom. Ensure your response contains only this question, with no additional commentary or elements. The entire response should be the question itself."
)
DOC_PROMPT = PromptTemplate.from_template(_doc_prompt_template)


def get_doc_chain(handler: BaseCallbackHandler = BaseCallbackHandler()) -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True, callbacks=[handler]),
        prompt=DOC_PROMPT,
        verbose=True,
        callbacks=[handler],
    )


if __name__ == "__main__":
    doc_chain = get_doc_chain()
    inputs = {"symptom": "abdominal cramps"}
    print(doc_chain(inputs))
