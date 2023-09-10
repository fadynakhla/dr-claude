from typing import Dict, List, Optional, Any, Callable
import asyncio
from pydantic import BaseModel
from loguru import logger
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.chains.base import Chain
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever

from dr_claude_old.retrieval.embeddings import (
    HuggingFaceEncoderEmbeddings,
    HuggingFaceEncoderEmbeddingsConfig,
)


class Symptom(BaseModel):
    symptom: str
    present: bool
    input_documents: Optional[List[Document]] = None


class SymptomList(BaseModel):
    symptoms: List[Symptom]


class MatchingChain(Chain):
    entity_extract_chain: LLMChain
    stuff_retrievals_match_chain: StuffDocumentsChain
    retriever: VectorStoreRetriever
    parser: Callable = parse_raw_extract

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = self.entity_extract_chain(inputs)
        symptom_list = self.parser(raw_symptom_extract["text"])
        logger.info("extracted symptom list: {}", symptom_list)
        for symptom in symptom_list.symptoms:  # suboptimal but fine for now
            symptom.input_documents = self.retriever.get_relevant_documents(
                symptom.symptom
            )
            logger.info(
                f"Retrieved {len(symptom.input_documents)} documents for {symptom.symptom}"
            )
            logger.debug(f"Retrieved documents: {symptom.input_documents}")
        return self.run_matching_loop(symptom_list)

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = await self.entity_extract_chain.acall(inputs)
        symptom_list = self.parser(raw_symptom_extract["text"])
        for symptom in symptom_list.symptoms:  # suboptimal but fine for now
            symptom.input_documents = await self.retriever.aget_relevant_documents(
                symptom.symptom
            )
        return self.run_matching_loop(symptom_list)

    def run_matching_batch(self, symptom_list: SymptomList) -> List[Dict[str, Any]]:
        async def run_batched(symptom_list: SymptomList) -> List[Dict[str, Any]]:
            tasks = []
            for symptom in symptom_list.symptoms:
                output = self.stuff_retrievals_match_chain.acall(dict(symptom))
                tasks.append(output)
            return await asyncio.gather(*tasks)

        return asyncio.run(run_batched(symptom_list))

    def run_matching_loop(self, symptom_list: SymptomList) -> List[Dict[str, Any]]:
        logger.info("matching symptom list {}", symptom_list)
        outputs = []
        for symptom in symptom_list.symptoms:
            output = self.stuff_retrievals_match_chain(dict(symptom))
            outputs.append(output)
        return outputs

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: List[Dict[str, str]],
        return_only_outputs: bool = False,
    ) -> List[Dict[str, str]]:
        new_outputs = []
        for output in outputs:
            new_output = super().prep_outputs(inputs, output)
            new_outputs.append(new_output)
        return new_outputs

    @property
    def input_keys(self) -> List[str]:
        return self.entity_extract_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["symptom_match", "present"]

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
            output_parser=XmlOutputParser(),
        )
        stuff_retrievals_match_chain = StuffDocumentsChain(
            llm_chain=symptom_match_chain,
            document_variable_name="retrievals",
            verbose=True,
            callbacks=[],
            output_key="symptom_match",
        )
        vectorstore = HuggingFAISS.from_model_config_and_texts(texts, retrieval_config)
        retriever = vectorstore.as_retriever()
        return cls(
            symptom_extract_chain=symptom_extract_chain,
            stuff_retrievals_match_chain=stuff_retrievals_match_chain,
            retriever=retriever,
        )

    @classmethod
    def from_anthropic_and_retriever(
        cls,
        symptom_extract_prompt: PromptTemplate,
        symptom_match_prompt: PromptTemplate,
        vec_store: Any,
        parser,
    ):
        llm = ChatAnthropic(temperature=0.0, verbose=True)
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
            output_key="entity_match",
        )
        return cls(
            symptom_extract_chain=symptom_extract_chain,
            stuff_retrievals_match_chain=stuff_retrievals_match_chain,
            retriever=vec_store.as_retriever(),
            parser=parser,
        )

    @classmethod
    def from_anthropic(
        cls,
        symptom_extract_prompt: PromptTemplate,
        symptom_match_prompt: PromptTemplate,
        retrieval_config: HuggingFaceEncoderEmbeddingsConfig,
        texts: List[str],
    ) -> "MatchingChain":
        anthropic = ChatAnthropic(temperature=0.0, verbose=True)
        return cls.from_llm(
            llm=anthropic,
            symptom_extract_prompt=symptom_extract_prompt,
            symptom_match_prompt=symptom_match_prompt,
            retrieval_config=retrieval_config,
            texts=texts,
        )


def parse_xml_line(line: str) -> str:
    try:
        root = ET.fromstring(line)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML line: {line}")
        raise e
    return root.text
