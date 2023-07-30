from typing import Union
import mcts
from loguru import logger

from dr_claude import kb_reading, datamodels, chaining_the_chains
from dr_claude.retrieval import retriever
from dr_claude.mcts import action_states


def main() -> None:
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    state = action_states.SimulationNextActionState(matrix, discount_rate=1e-9)
    for symptom_group in kb.condition_symptoms.values():
        for s in symptom_group:
            if s.name.lower().strip() == "fever":
                symptom = datamodels.SymptomTransformer.to_symptom(s)
    state.pertinent_pos.update(symptom)
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    searcher = mcts.mcts(timeLimit=3000, rolloutPolicy=rollout_policy)
    note = ("The patient has syncope, vertigo, nausea and is sweating",)
    symptoms = [
        "fever",
        "vertigo",
        "syncope",
        "nausea",
        "heavy sweating",
        "abdominal cramps",
    ]
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path="bert-base-uncased",
        device="cpu",
    )
    chain_chainer = chaining_the_chains.ChainChainer(
        patient_note=note,
        retrieval_config=retrieval_config,
        symptoms=symptoms,
    )
    action = None
    while not isinstance(
        (action := searcher.search(initialState=state)), datamodels.Condition
    ):
        assert isinstance(action, datamodels.Symptom)
        logger.info(f"{action=}")
        patient_symptom_response = chain_chainer.interaction(action.name)
    diagnosis = action
    logger.info(f"Diagnosis: {diagnosis}")
    print(chain_chainer.interaction("fever"))


if __name__ == "__main__":
    main()
