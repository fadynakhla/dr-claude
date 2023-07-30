from typing import Optional, Union, Dict, Any, List
from uuid import UUID
from langchain.schema.output import LLMResult
from starlette.templating import _TemplateResponse
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

import mcts
from langchain.callbacks.base import AsyncCallbackHandler

from loguru import logger

from dr_claude import kb_reading, datamodels, chaining_the_chains
from dr_claude.retrieval import retriever
from dr_claude.claude_mcts import action_states, multi_choice_mcts
from dr_claude.chains import decision_claude, matcher, prompts, doctor, patient


from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

import json
from loguru import logger

app = FastAPI()


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )
class PatientHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        text = response.generations[0][0].text
        self.websocket.send_json({"patient": text})


class DoctorHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        text = response.generations[0][0].text
        self.websocket.send_json({"doctor": text})


@app.on_event("startup")
async def startup_event() -> None:
    global kb
    global chain_chainer
    global symptom_name_to_symptom
    global fever
    global matcher_chain
    global action_picker

    embedding_model_name = "/data/models/RoSaBERTa_large/"
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path=embedding_model_name,
        device="cpu",
    )
    symptom_name_to_symptom: Dict[str, datamodels.Symptom] = {}
    for symptom_group in kb.condition_symptoms.values():
        for weighted_symptom in symptom_group:
            symptom = datamodels.SymptomTransformer.to_symptom(weighted_symptom)
            symptom_name_to_symptom[symptom.name.strip()] = symptom
            if weighted_symptom.name.lower().strip() == "fever":
                fever = symptom
    matcher_chain = matcher.MatchingChain.from_anthropic(
        symptom_extract_prompt=prompts.SYMPTOM_EXTRACT_PROMPT,
        symptom_match_prompt=prompts.SYMPTOM_MATCH_PROMPT,
        retrieval_config=retrieval_config,
        texts=list(set(symptom_name_to_symptom)),
    )
    action_picker = decision_claude.DecisionClaude()


@app.get("/health")
def health():
    return {"status": "up"}


@app.get("/")
async def get():
    return {"status": "up"}


@app.websocket("/dr_claude")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    doc_handler = DoctorHandler(websocket)
    patient_handler = PatientHandler(websocket)
    doc_chain = doctor.get_doc_chain(doc_handler)
    patient_chain = patient.get_patient_chain(patient_handler)
    chainer = chaining_the_chains.ChainChainer(matcher_chain, doc_chain, patient_chain)
    while True:
        try:
            message = await receive_message(websocket)
            logger.info("Received {}", message)
            await run_chain(note=message["content"], chainer=chainer)
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
            break
        except KeyboardInterrupt:
            logger.info("keyboard interrupt")
            break
        except RuntimeError:
            logger.info("websocket disconnect")
            break
        except ValueError:
            continue
        except Exception as e:
            logger.error(e)


async def receive_message(websocket: WebSocket):
    try:
        unparsed_message = await websocket.receive()
    except RuntimeError:
        raise

    try:
        message_text: str = unparsed_message["text"]
        message_dict = json.loads(message_text)
    except Exception:
        raise ValueError("Failed to parse message")
    return message_dict


async def run_chain(note: str, chainer: chaining_the_chains.ChainChainer):
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    state = action_states.SimulationNextActionState(matrix, discount_rate=1e-9)

    state.pertinent_pos.update([fever])
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    searcher = multi_choice_mcts.MultiChoiceMCTS(
        timeLimit=3000, rolloutPolicy=rollout_policy
    )

    while not isinstance(
        (actions := searcher.search(initialState=state, top_k=5))[0],
        datamodels.Condition,
    ):
        assert isinstance(actions[0], datamodels.Symptom)
        logger.info(f"{actions=}")

        action_name = action_picker(actions=actions, state=state)

        patient_symptom_response = chainer.interaction(note, action_name)
        new_positives, new_negatives = make_new_symptoms(
            action_name, patient_symptom_response
        )
        state.pertinent_pos.update(new_positives)
        state.pertinent_neg.update(new_negatives)

    diagnosis = actions[0]
    logger.info(f"Diagnosis: {diagnosis}")


def make_new_symptoms(
    action_name: str, patient_symptom_response: List[datamodels.SymptomMatch]
):
    non_key_error_symptoms = []
    for s in patient_symptom_response:
        try:
            symptom_name = s.symptom_match.strip()
            symptom_name_to_symptom[symptom_name]
            non_key_error_symptoms.append(symptom_name)
        except KeyError:
            pass

    if non_key_error_symptoms:
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
    else:
        new_positives = []
        new_negatives = [symptom_name_to_symptom[action_name.strip()]]
    return new_positives, new_negatives
