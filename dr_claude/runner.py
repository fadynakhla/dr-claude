from typing import List, Union, Dict
import mcts
from loguru import logger

from dr_claude import kb_reading, datamodels, chaining_the_chains
from dr_claude.retrieval import retriever
from dr_claude.claude_mcts import action_states, multi_choice_mcts
from dr_claude.chains import decision_claude


def main() -> None:
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
    searcher = multi_choice_mcts.MultiChoiceMCTS(timeLimit=3000, rolloutPolicy=rollout_policy)

    action_picker = decision_claude.get_decision_claude()
    note = ("The patient has syncope, vertigo, nausea and is sweating",)
    # embedding_model_name = "/data/models/RoSaBERTa_large/"
    embedding_model_name = "bert-base-uncased"
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path=embedding_model_name,
        device="cpu",
    )
    chain_chainer = chaining_the_chains.ChainChainer(
        patient_note=note,
        retrieval_config=retrieval_config,
        symptoms=list(set(symptom_name_to_symptom)),
    )
    while not isinstance(
        (actions := searcher.search(initialState=state, top_k=5))[0], datamodels.Condition
    ):
        assert isinstance(actions[0], datamodels.Symptom)
        logger.info(f"{actions=}")
        actions = [action for action in actions if valid_action(action, state)]

        action_picker_inputs = get_action_picker_inputs(actions, state)
        action_name = action_picker(action_picker_inputs)["text"]

        patient_symptom_response = chain_chainer.interaction(action_name)
        exit()
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

    action = actions[0]
    diagnosis = action
    logger.info(f"Diagnosis: {diagnosis}")
    print(chain_chainer.interaction("fever"))


def valid_action(
    action: Union[datamodels.Condition, datamodels.Symptom],
    state: action_states.SimulationNextActionState
) -> bool:
    return (
        not isinstance(action, datamodels.Condition)
        and action not in state.pertinent_pos
        and action not in state.pertinent_neg
    )


def get_action_picker_inputs(actions: List[datamodels.Symptom], state) -> Dict[str, str]:
    return {
        "positive_symptoms": " | ".join([action.name for action in state.pertinent_pos]),
        "negative_symptoms": " | ".join([action.name for action in state.pertinent_neg]),
        "symptoms": " | ".join([action.name for action in actions]),
    }


if __name__ == "__main__":
    main()
