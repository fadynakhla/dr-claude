from typing import Union, Dict
import mcts
from loguru import logger

from dr_claude import kb_reading, datamodels, chaining_the_chains
from dr_claude.retrieval import retriever
from dr_claude.claude_mcts import action_states


def a():
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    state = action_states.SimulationNextActionState(matrix, discount_rate=1e-9)
    symptom_name_to_symptom: Dict[str, datamodels.Symptom] = {}
    for symptom_group in kb.condition_symptoms.values():
        for weighted_symptom in symptom_group:
            symptom = datamodels.SymptomTransformer.to_symptom(weighted_symptom)
            symptom_name_to_symptom[symptom.name.strip()] = symptom
            if weighted_symptom.name.lower().strip() == "fever":
                fever = symptom
    state.pertinent_pos.update([fever])
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    searcher = mcts.mcts(timeLimit=3000, rolloutPolicy=rollout_policy)
    note = ("The patient has syncope, vertigo, nausea and is sweating",)
    embedding_model_name = "/data/models/RoSaBERTa_large/"
    # embedding_model_name = "bert-base-uncased"
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path=embedding_model_name,
        device="cpu",
    )
    chain_chainer = chaining_the_chains.ChainChainer(
        retrieval_config=retrieval_config,
        symptoms=list(set(symptom_name_to_symptom)),
    )
    while not isinstance(
        (action := searcher.search(initialState=state)), datamodels.Condition
    ):
        assert isinstance(action, datamodels.Symptom)
        logger.info(f"{action=}")
        patient_symptom_response = chain_chainer.interaction(note, action.name)
        new_positives = [
            symptom_name_to_symptom[s.symptom_match.strip()]
            for s in patient_symptom_response
            if s.present
        ]
        new_negatives = [
            symptom_name_to_symptom[s.symptom_match.strip()]
            for s in patient_symptom_response
            if not s.present
        ]
        state.pertinent_pos.update(new_positives)
        state.pertinent_neg.update(new_negatives)

    diagnosis = action
    logger.info(f"Diagnosis: {diagnosis}")
    print(chain_chainer.interaction("fever"))


if __name__ == "__main__":
    main()
