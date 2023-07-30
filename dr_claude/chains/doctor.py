from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_doc_prompt_template = (
    "You are an inteligent and curious doctor. You are interacting with a patient and "
    "you wish to ask them whether they are experiencing the following symptom: {symptom}.\n\n"
    "Remember to only ask one question at a time about the provided symptom."
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
