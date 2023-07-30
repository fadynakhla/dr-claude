from typing import Any, Dict, List
from dr_claude.retrieval.retriever import HuggingFAISS
from dr_claude import datamodels
from langchain import LLMChain

from dr_claude.chains import doctor
from dr_claude.chains import patient
from dr_claude.chains import matcher
from dr_claude.chains import prompts
from dr_claude.retrieval import retriever
from loguru import logger
import time


class ChainChainer:
    def __init__(
        self,
        matcher_chain: matcher.MatchingChain,
        doc_chain: LLMChain,
        patient_chain: LLMChain,
    ) -> None:
        self.doc_chain = doc_chain
        self.patient_chain = patient_chain
        self.matcher_chain = matcher_chain

    def interaction(
        self, patient_note: str, symptom: str
    ) -> List[datamodels.SymptomMatch]:
        doc_inputs = {"symptom": symptom}
        doc_response = self.doc_chain(doc_inputs)["text"]
        time.sleep(0.5)
        logger.info("Doc question: {}", doc_response)
        patient_inputs = {"medical_note": patient_note, "question": doc_response}
        patient_response = self.patient_chain(patient_inputs)["text"]
        time.sleep(0.5)
        logger.info("Patient response: {}", patient_response)
        matcher_inputs = {"question": doc_response, "response": patient_response}
        match_list = self.matcher_chain(matcher_inputs)
        return [datamodels.SymptomMatch(**match) for match in match_list]


if __name__ == "__main__":
    note = "Patient presents with abdominal cramps and nausea. He does not have a fever, but is feeling dizzy."
    symptoms = [
        "fever",
        "cough",
        "headache",
        "sore throat",
        "runny nose",
        "abdominal cramps",
    ]
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path="bert-base-uncased",
        device="cpu",
    )
    chain_chainer = ChainChainer(
        patient_note=note,
        retrieval_config=retrieval_config,
        symptoms=symptoms,
    )
    print(chain_chainer.interaction("abdominal cramps"))
