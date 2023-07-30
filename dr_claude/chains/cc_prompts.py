from langchain.prompts import PromptTemplate

_symptom_extract_template = """Consider the following conversation patient note:
Patient note: {note}

Choose on of the symptoms to be the chief complaint (it is usually the first symptom mentioned).

Provide your response strictly in the following format, replacing only the name_of_chief_complaint (keeping : yes), and refrain from including any additional text:
<symptom> name_of_chief_complaint </symptom>
"""

_symptom_match_template = """Given the symptom: {symptom} which of the following retrievals is the best match?
Retrievals:
{retrievals}

Select only one and write it below in the following format:
<match> choice </match>

Remember, do not include any other text, ensure your choice is in the provided retrievals, and follow the output format.
"""


CC_EXTRACT_PROMPT = PromptTemplate.from_template(_symptom_extract_template)
CC_MATCH_PROMPT = PromptTemplate.from_template(_symptom_match_template)
