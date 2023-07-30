from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import asyncio
from abc import abstractmethod
import pandas as pd
from tqdm import tqdm
from xml.etree.ElementTree import ParseError
from langchain.llms import Anthropic
from langchain import LLMChain, PromptTemplate
from langchain.schema import BaseOutputParser

from dr_claude import datamodels

DEFAULT_FREQ_TERM_TO_WEIGHT = {
    "Very common": 0.9,
    "Common": 0.6,
    "Uncommon": 0.3,
    "Rare": 0.1,
}


class WeightedSymptomXMLOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a list.

    Args:
        frequency_term_to_weight: Mapping from a frequncy term to the
            causal weight that it constitutes.
    """

    frequency_term_to_weight: Dict[str, float] = DEFAULT_FREQ_TERM_TO_WEIGHT

    @property
    def _type(self) -> str:
        return "xml"

    def parse(self, text: str) -> List[datamodels.WeightedSymptom]:
        """Parse the output of an LLM call."""
        root = ET.fromstring(text)
        symptoms = []
        for symptom_elem in root:
            name = symptom_elem.find("name").text
            frequency = symptom_elem.find("frequency").text
            weight = self.frequency_term_to_weight.get(frequency, self.min_weight)
            symptom = datamodels.WeightedSymptom(
                umls_code="none", name=name, weight=weight
            )
            symptoms.append(symptom)
        return symptoms

    @property
    def min_weight(self) -> float:
        return min(self.frequency_term_to_weight.values())


def kb_to_dataframe(kb: datamodels.DiseaseSymptomKnowledgeBase) -> pd.DataFrame:
    rows: List[Tuple[str, str, str, str]] = []
    cols = ("Disease Code", "Disease", "Symptom Code", "Symptom", "Weight", "Noise")
    for condition, symptoms in kb.condition_symptoms.items():
        for s in symptoms:
            rows.append(
                (
                    condition.umls_code,
                    condition.name,
                    s.umls_code,
                    s.name,
                    s.weight,
                    s.noise_rate,
                )
            )
    return pd.DataFrame(rows, columns=cols)


async def get_updated_weights(
    condition_symptoms: Dict[datamodels.Condition, List[datamodels.WeightedSymptom]],
    llm_chain: LLMChain,
) -> Dict[datamodels.Condition, List[datamodels.WeightedSymptom]]:
    async def get_symptom_weights(
        sem: asyncio.Semaphore,
        condition: datamodels.Condition,
        symptoms: datamodels.WeightedSymptom,
    ):
        async with sem:
            symptoms_str = ", ".join([s.name for s in symptoms])
            try:
                result = await llm_chain.arun(
                    condition=condition.name, symptoms_list=symptoms_str
                )
            except ParseError:
                return None
            return (condition, result)

    sem = asyncio.Semaphore(1)  # max concurrent calls
    weight_updated_condition_symptoms = {}
    tasks = []
    for condition, symptoms in condition_symptoms.items():
        task = asyncio.ensure_future(get_symptom_weights(sem, condition, symptoms))
        tasks.append(task)

    with tqdm(total=len(condition_symptoms)) as progress:
        for f in asyncio.as_completed(tasks):
            outcome = await f
            if outcome is not None:
                condition, result = outcome
                weight_updated_condition_symptoms[condition] = result
            progress.update(1)

        await asyncio.gather(*tasks)
        return weight_updated_condition_symptoms


def main() -> None:
    from dr_claude import kb_reading

    llm = Anthropic(model="claude-2", temperature=0.0, max_tokens_to_sample=2000)
    prompt_template = """Here is a list of symptoms for the condition {condition}.
                        Symptoms: {symptoms_list}.

                        Here is the output schema:
                        <?xml version="1.0" encoding="UTF-8"?>
                            <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
                            
                            <xs:element name="covidSymptoms">
                                <xs:complexType>
                                <xs:sequence>
                                    <xs:element name="symptom" maxOccurs="unbounded">
                                    <xs:complexType>
                                        <xs:sequence>
                                        <xs:element name="name" type="xs:string"/>
                                        <xs:element name="frequency">
                                            <xs:simpleType>
                                            <xs:restriction base="xs:string">
                                                <xs:enumeration value="Very common"/>
                                                <xs:enumeration value="Common"/> 
                                                <xs:enumeration value="Uncommon"/>
                                                <xs:enumeration value="Rare"/>
                                            </xs:restriction>
                                            </xs:simpleType>
                                        </xs:element>
                                        </xs:sequence>
                                    </xs:complexType>
                                    </xs:element>
                                </xs:sequence>
                                </xs:complexType>
                            </xs:element>
                            
                            </xs:schema>
                Please parse the symptoms into the above schema, assigning a correct frequency value to each symptom.
                """
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(
        llm=llm, prompt=prompt, output_parser=WeightedSymptomXMLOutputParser()
    )
    reader = kb_reading.NYPHKnowldegeBaseReader("data/NYPHKnowldegeBase.html")
    kb = reader.load_knowledge_base()
    final_result = asyncio.run(get_updated_weights(kb.condition_symptoms, llm_chain))
    weight_updated_kb = datamodels.DiseaseSymptomKnowledgeBase(
        condition_symptoms=final_result
    )
    kb_to_dataframe(weight_updated_kb).to_csv("data/ClaudeKnowledgeBase-1.csv")


if __name__ == "__main__":
    main()
