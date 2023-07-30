from langchain.prompts import PromptTemplate


_symptom_extract_template = """Given the following conversation:
Question: {question}
Response: {response}

Please write out the medical symptoms that appear as well as whether they are present.

Your response should be in the following format please do not include any other text:
<symptom> symptom1 : yes </symptom>
<symptom> symptom2 : no </symptom>
"""

_symptom_match_template = """Given the symptom: {symptom} which of the following retrievals is the best match?
Retrievals:
{retrievals}

Select only one and write it below in the following formatt:
<match> match </match>

Remember, do not include any other text and ensure your choice is in the provided retrievals.
"""


SYMPTOM_EXTRACT_PROMPT = PromptTemplate.from_template(_symptom_extract_template)
SYMPTOM_MATCH_PROMPT = PromptTemplate.from_template(_symptom_match_template)
