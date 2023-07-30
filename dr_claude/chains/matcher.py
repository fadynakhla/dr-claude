from typing import Any, Dict, List, Optional
import asyncio
from langchain import Anthropic, OpenAI
import pydantic
from langchain.chains.base import Chain
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever

from dr_claude.retrieval.retriever import HuggingFAISS
from dr_claude.retrieval.embeddings import HuggingFaceEncoderEmbeddingsConfig


class Symptom(pydantic.BaseModel):
    symptom: str
    present: bool
    retrievals: Optional[List[Document]] = None


class SymptomList(pydantic.BaseModel):
    symptoms: List[Symptom]


class MatchingChain(Chain):

    symptom_extract_chain: LLMChain
    stuff_retrievals_match_chain: StuffDocumentsChain
    retriever: VectorStoreRetriever

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = self.symptom_extract_chain(inputs)
        symptom_list = parse_raw_extract(raw_symptom_extract["text"])
        for symptom in symptom_list.symptoms: # suboptimal but fine for now
            symptom.retrievals = self.retriever.get_relevant_documents(symptom.symptom)
        return self.run_matching_batch(symptom_list)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = await self.symptom_extract_chain.acall(inputs)
        symptom_list = parse_raw_extract(raw_symptom_extract["text"])
        for symptom in symptom_list.symptoms: # suboptimal but fine for now
            symptom.retrievals = await self.retriever.aget_relevant_documents(symptom.symptom)
        return self.run_matching_batch(symptom_list)

    def run_matching_batch(self, symptom_list: SymptomList) -> List[Dict[str, Any]]:

        async def run_batched(symptom_list: SymptomList) -> List[Dict[str, Any]]:
            tasks = []
            for symptom in symptom_list.symptoms:
                output = self.stuff_retrievals_match_chain.acall(symptom.dict(exclude={"present"}))
                output["present"] = symptom.present
                tasks.append(output)
            return await asyncio.gather(*tasks)

        return asyncio.run(run_batched(symptom_list))

    @property
    def input_keys(self) -> List[str]:
        return self.symptom_extract_chain.input_keys

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        symptom_extract_prompt: PromptTemplate,
        symptom_match_prompt: PromptTemplate,
        retrieval_config: HuggingFaceEncoderEmbeddingsConfig,
        texts: List[str],
    ) -> "MatchingChain":
        symptom_extract_chain = LLMChain(
            llm=llm,
            prompt=symptom_extract_prompt,
        )
        symptom_match_chain = LLMChain(
            llm=llm,
            prompt=symptom_match_prompt,
        )
        stuff_retrievals_match_chain = StuffDocumentsChain(
            llm_chain=symptom_match_chain,
            document_variable_name="retrievals",
            verbose=True,
            callbacks=[],
        )
        vectorstore = HuggingFAISS.from_model_config_and_texts(texts, retrieval_config)
        retriever = vectorstore.as_retriever()
        return cls(
            symptom_extract_chain=symptom_extract_chain,
            stuff_retrievals_match_chain=stuff_retrievals_match_chain,
            retriever=retriever,
        )

    @classmethod
    def from_anthropic(
        cls,
        symptom_extract_prompt: PromptTemplate,
        symptom_match_prompt: PromptTemplate,
        retrieval_config: HuggingFaceEncoderEmbeddingsConfig,
        texts: List[str],
    ) -> "MatchingChain":
        anthropic = Anthropic(
            temperature=0.1,
            verbose=True,
        )
        return cls.from_llm(
            llm=anthropic,
            symptom_extract_prompt=symptom_extract_prompt,
            symptom_match_prompt=symptom_match_prompt,
            retrieval_config=retrieval_config,
            texts=texts,
        )

def parse_raw_extract(text: str) -> SymptomList:
    symptom_strings = text.split("\n")
    symptoms = []
    for symptom_string in symptom_strings:
        name, present = symptom_string.split(":")
        symptom = Symptom(symptom=name.strip(), present=present.strip() == "yes")
        symptoms.append(symptom)
    return SymptomList(symptoms=symptoms)
