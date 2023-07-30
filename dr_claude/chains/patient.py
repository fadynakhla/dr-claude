from langchain.chat_models import ChatAnthropic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# _patient_prompt_template = (
#     "You are a patient who is seeing their doctor. Your medical state is described as follows:\n\n"
#     "{medical_note}\n\n"
#     "The doctor will ask you questions about your symptoms. Answer them only according to the "
#     "information provided above. If the doctor asks if you have a symptom not "
#     "mentioned above, answer no and do not reveal any other information.\n\n"
#     "Question:\n{question}\n\n"

#     "Remember to only answer one question at a time and to only answer according to the information provided above.\n\n"
#     "Answer:"
# )
_patient_prompt_template = (
    "You are a patient who is visiting their doctor. Your medical condition is detailed as follows:\n\n"
    "{medical_note}\n\n"
    "The doctor will ask you questions about your symptoms. It's important to answer each question one at a time and strictly according to the "
    "information mentioned above. If the doctor inquires about a symptom not "
    "discussed in the above note, your response should be 'no'. Additionally, avoid volunteering any other information unless specifically asked.\n\n"
    "Question:\n{question}\n\n"
    "Please remember that your answers should be focused solely on the specific question asked, without reference to other symptoms or information not requested.\n\n"
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
