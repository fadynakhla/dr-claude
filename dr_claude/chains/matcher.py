from typing import Any, Dict, List, Optional
import asyncio
import pydantic
from langchain.chains.base import Chain
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain.llms import Anthropic
from langchain.llms.base import LLM
from langchain.schema import BaseOutputParser
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate

from dr_claude.retrieval.retriever import HuggingFAISS
from dr_claude.retrieval.embeddings import HuggingFaceEncoderEmbeddingsConfig


class Symptom(pydantic.BaseModel):
    symptom: str
    present: bool
    retrievals: Optional[List[str]] = None


class SymptomList(pydantic.BaseModel):
    symptoms: List[Symptom]


class MatchingChain(Chain):

    def __init__(
        self,
        llm: LLM,
        symptom_extract_prompt: PromptTemplate,
        symptom_match_prompt: PromptTemplate,
        vectorstore: HuggingFAISS
    ) -> None:
        self.symptom_extract_chain = LLMChain(
            llm=llm,
            prompt=symptom_extract_prompt,
        )
        symptom_match_chain = LLMChain(
            llm=llm,
            prompt=symptom_match_prompt,
        )
        self.stuff_retrievals_match_chain = StuffDocumentsChain(
            llm_chain=symptom_match_chain,
            document_variable_name="retrievals",
            verbose=True,
            callbacks=[],
        )
        self.retriever = vectorstore.as_retriever()

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = self.symptom_extract_chain(inputs)
        symptom_list = parse_raw_extract(raw_symptom_extract["text"])
        # asyncio.run(self.retriever.aget_relevant_documents([s.symptom for s in symptoms]))
        for symptom in symptom_list.symptoms:
            symptom.retrievals = self.retriever.get_relevant_documents(symptom.symptom)


    def run_matching_batch(self, symptom_list: SymptomList) -> List[Dict[str, Any]]:

        async def run_batched(symptom_list: SymptomList) -> List[Dict[str, Any]]:
            tasks = []
            for symptom in symptom_list.symptoms:
                output = self.stuff_retrievals_match_chain.acall(symptom)
                tasks.append(output)
            return await asyncio.gather(*tasks)

        return asyncio.run(run_batched(symptom_list))

    @property
    def input_keys(self) -> List[str]:
        return self.symptom_extract_chain.input_keys


def parse_raw_extract(text: str) -> SymptomList:
    symptom_strings = text.split("\n")
    symptoms = []
    for symptom_string in symptom_strings:
        name, present = symptom_string.split(":")
        symptom = Symptom(symptom=name.strip(), present=present.strip() == "yes")
        symptoms.append(symptom)
    return SymptomList(symptoms=symptoms)
