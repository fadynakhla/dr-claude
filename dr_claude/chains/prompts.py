from langchain.prompts import PromptTemplate


# _symptom_extract_template = """Given the following conversation:
# Question: {question}
# Response: {response}

# Please write out the medical symptoms that appear as well as whether they are present.

# Your response should be in the following format please do not include any other text:
# <symptom> symptom1 : yes </symptom>
# <symptom> symptom2 : no </symptom>

# Remember, do not include any other text, ensure your choices are in the provided conversation, and follow the output format.
# """
_symptom_extract_template = """Consider the following conversation:
Question: {question}
Response: {response}

Identify the mentioned medical symptoms, and note whether they are present or not.

Provide your responses strictly in the following format, and refrain from including any additional text:
<symptom> symptom1 : yes </symptom>
<symptom> symptom2 : no </symptom>

Please remember, your response should only contain the identified symptoms from the given conversation and their presence status, in the specified format. Refrain from adding any additional details or text.
"""

_symptom_match_template = """Given the symptom: {symptom} which of the following retrievals is the best match?
Retrievals:
{retrievals}

Select only one and write it below in the following format:
<match> match </match>

Remember, do not include any other text, ensure your choice is in the provided retrievals, and follow the output format.
"""


SYMPTOM_EXTRACT_PROMPT = PromptTemplate.from_template(_symptom_extract_template)
SYMPTOM_MATCH_PROMPT = PromptTemplate.from_template(_symptom_match_template)
