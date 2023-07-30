from typing import Any, Dict, List, Optional, Callable
import asyncio
import xml.etree.ElementTree as ET

import pydantic
import loguru
from langchain.chat_models import ChatAnthropic
from langchain.chains.base import Chain
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema.output_parser import BaseOutputParser

from dr_claude.retrieval.retriever import HuggingFAISS
from dr_claude.retrieval.embeddings import HuggingFaceEncoderEmbeddingsConfig


logger = loguru.logger


class Symptom(pydantic.BaseModel):
    symptom: str
    present: bool
    input_documents: Optional[List[Document]] = None


class SymptomList(pydantic.BaseModel):
    symptoms: List[Symptom]


def parse_raw_extract(text: str) -> SymptomList:
    symptom_strings = text.strip().split("\n")
    symptoms = []
    logger.debug(f"Raw symptom strings: {symptom_strings}")
    for symptom_string in symptom_strings:
        logger.debug(f"Single line response: {symptom_string}")
        symptom_string = parse_xml_line(symptom_string)
        # gets here
        name, present = symptom_string.split(":")
        symptom = Symptom(symptom=name.strip(), present=present.strip() == "yes")
        symptoms.append(symptom)
    logger.info("finished parsing")
    return SymptomList(symptoms=symptoms)


def parse_raw_extract_cc(text: str) -> SymptomList:
    symptom_strings = text.strip().split("\n")
    symptoms = []
    logger.debug(f"Raw symptom strings: {symptom_strings}")
    for symptom_string in symptom_strings:
        logger.debug(f"Single line response: {symptom_string}")
        symptom_string = parse_xml_line(symptom_string)
        name = symptom_string
        symptom = Symptom(symptom=name.strip(), present="yes")
        symptoms.append(symptom)
    logger.info("finished parsing")
    return SymptomList(symptoms=symptoms)


class Symptom(pydantic.BaseModel):
    symptom: str
    present: bool
    input_documents: Optional[List[Document]] = None


class SymptomList(pydantic.BaseModel):
    symptoms: List[Symptom]


class XmlOutputParser(BaseOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string.."""

    @property
    def lc_serializable(self) -> bool:
        """Whether the class LangChain serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    def parse(self, text: str) -> str:
        """Returns the input text with no changes."""
        return parse_xml_line(text.strip())


class MatchingChain(Chain):
    symptom_extract_chain: LLMChain
    stuff_retrievals_match_chain: StuffDocumentsChain
    retriever: VectorStoreRetriever
    parser: Callable = parse_raw_extract

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_symptom_extract = self.symptom_extract_chain(inputs)
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
        raw_symptom_extract = await self.symptom_extract_chain.acall(inputs)
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

    # def _validate_outputs(self, outputs: List[Dict[str, Any]]) -> None:
    #     for output in outputs:
    #         super()._validate_outputs(output)

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
        return self.symptom_extract_chain.input_keys

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
        vec_store: HuggingFAISS,
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
            output_parser=XmlOutputParser(),
        )
        stuff_retrievals_match_chain = StuffDocumentsChain(
            llm_chain=symptom_match_chain,
            document_variable_name="retrievals",
            verbose=True,
            callbacks=[],
            output_key="symptom_match",
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


if __name__ == "__main__":
    from dr_claude.chains import prompts

    chain = MatchingChain.from_anthropic(
        symptom_extract_prompt=prompts.SYMPTOM_EXTRACT_PROMPT,
        symptom_match_prompt=prompts.SYMPTOM_MATCH_PROMPT,
        retrieval_config=HuggingFaceEncoderEmbeddingsConfig(
            model_name_or_path="bert-base-uncased",
            device="cpu",
        ),
        texts=["fever", "cough", "headache", "sore throat", "runny nose"],
    )
    inputs = {
        "question": "Do you have a fever?",
        "response": "yes and I have a headache as well",
    }
    print(chain(inputs))
