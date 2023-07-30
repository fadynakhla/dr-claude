from langchain.chat_models import ChatAnthropic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


_patient_prompt_template = (
    "You are a patient who is seeing their doctor. Your medical state is described as follows:\n\n"
    "{medical_note}\n\n"
    "The doctor will ask you questions about your symptoms. Answer them only according to the "
    "information provided above. If the doctor asks if you have a symptom not "
    "mentioned above, answer no and do not reveal any other symptoms.\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)
PATIENT_PROMPT = PromptTemplate.from_template(_patient_prompt_template)


def get_patient_chain(handler: BaseCallbackHandler = BaseCallbackHandler()) -> LLMChain:
    return LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True, callbacks=[handler]),
        prompt=PATIENT_PROMPT,
        verbose=True,
        callbacks=[handler],
    )


if __name__ == "__main__":
    patient_chain = get_patient_chain()
    note = "Patient presents with abdominal cramps and nausea. He does not have a fever, but is feeling dizzy."
    question = "Are you experiencing any abdominal cramps?"
    inputs = {"medical_note": note, "question": question}
    print(patient_chain(inputs))
