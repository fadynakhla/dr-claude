"""This module contains logic for generating a chief complaint for the patient."""

import xml.etree.ElementTree as ET
from pydantic import BaseModel, validator
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

_symptom_extract_template = """Consider the following conversation patient note:
Patient note: {note}

Choose on of the symptoms to be the chief complaint (it is usually the first symptom mentioned). Output the chief complaint following the schema:
<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xs:element name="ChiefComplaint" type="xs:string"/>

</xs:schema>
"""

_symptom_match_template = """Given the symptom: '{symptom}' which of the following retrievals is the best match?
Retrievals:
{retrievals}

Select only one and write it below in the following format:
<match> choice </match>

Remember, do not include any other text, ensure your choice is in the provided retrievals, and follow the output format.
"""
CC_EXTRACT_PROMPT = PromptTemplate.from_template(_symptom_extract_template)
CC_MATCH_PROMPT = PromptTemplate.from_template(_symptom_match_template)

_RETRIEVAL_XML_ENUM_PROMPT_PREFIX = (
    """Here is the XML schema for some relevant clinical concepts:"""
    """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xs:element name="ClinicalConcept" type="ConceptType"/>

  <xs:simpleType name="ConceptType">
    <xs:restriction base="xs:string">
"""
)
_RETRIEVAL_XML_ENUM_PROMPT_SUFFIX = """    </xs:restriction>
  </xs:simpleType>

</xs:schema>

Out of the available options specified above, which of the concepts is most similar (or matches) to: '{query}'? Please just output the answer without additional comments or annotation."""


class RetrievalXMLEnumPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes a list of retrievals and a query (e.g. a symptom) as input,
    formats the prompt template to provide the retrievals as XML enum options in the schema.
    """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        assert "retrievals" in v
        assert "query" in v
        return v

    def format(self, **kwargs) -> str:
        retrievals = kwargs["retrievals"]
        query = kwargs["query"]
        options = "\n".join([f'    <xs:enumeration value="{r}"/>' for r in retrievals])
        return f"{_RETRIEVAL_XML_ENUM_PROMPT_PREFIX}{options}\n{_RETRIEVAL_XML_ENUM_PROMPT_SUFFIX.format(query=query)}"

    def _prompt_type(self) -> str:
        return "retrieval-xml-enumer"


class ChiefComplaintXMLParser(BaseOutputParser[str]):
    """Parse the output of the chief complaint XML."""

    @property
    def _type(self) -> str:
        return "xml"

    def parse(self, text: str) -> str:
        """Extract the chief complaint from the string."""
        root = ET.fromstring(text.strip())
        chief_complaint_element = root.find(
            ".//xs:element[@name='ChiefComplaint']",
            namespaces={"xs": "http://www.w3.org/2001/XMLSchema"},
        )
        assert (
            chief_complaint_element is not None
        ), f"Model did not output according to the expected schema, got: {text}"
        assert (
            chief_complaint_element.text is not None
        ), f"Model did not output according to the expected schema, got: {text}"
        return chief_complaint_element.text


def get_chief_complaint_chain() -> LLMChain:
    cc_chooser = LLMChain(
        llm=ChatAnthropic(temperature=0.0, verbose=True, callbacks=[]),
        prompt=CC_EXTRACT_PROMPT,
        verbose=True,
        callbacks=[],
        output_parser=ChiefComplaintXMLParser(),
    )
    return cc_chooser


if __name__ == "__main__":
    templ = RetrievalXMLEnumPromptTemplate(input_variables=["query", "retrievals"])
    p = templ.format(query="fever", retrievals=["fever", "cough", "emesis"])
    llm = ChatAnthropic(temperature=0.0, verbose=True, callbacks=[])
    print(llm.predict(p))
