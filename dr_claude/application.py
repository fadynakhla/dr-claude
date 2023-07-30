from typing import Optional, Tuple, Union, Dict, Any, List
import asyncio
import nest_asyncio
from uuid import UUID
from langchain.schema.output import LLMResult
from starlette.templating import _TemplateResponse
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

import mcts
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler

from loguru import logger

from dr_claude import kb_reading, datamodels, chaining_the_chains
from dr_claude.retrieval import retriever
from dr_claude.claude_mcts import action_states, multi_choice_mcts
from dr_claude.chains import (
    decision_claude,
    matcher,
    prompts,
    doctor,
    patient,
    cc_prompts,
)


from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from dr_claude.retrieval.retriever import HuggingFAISS
import json
from loguru import logger

import transformers

transformers.set_seed(42)

app = FastAPI()


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

nest_asyncio.apply()


class PatientHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.error("PATIENT ON LLM END")
        text = response.generations[0][0].text
        # await self.websocket.send_json({"patient": text})
        asyncio.run(self.websocket.send_json({"patient": text}))


def send_json_sync(websocket, text, patient: bool):
    # Define an asynchronous function that we'll run in the event loop
    async def async_send_json(websocket, text):
        await websocket.send_json({"patient": text})

    async def async_send_doc_json(websocket, text):
        await websocket.send_json({"doctor": text})

    # Get a reference to the current event loop, or create a new one
    loop = asyncio.get_event_loop()

    # Run the async function in the event loop
    if patient:
        loop.run_until_complete(async_send_json(websocket, text))
    else:
        loop.run_until_complete(async_send_doc_json(websocket, text))


class DoctorHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.error("DOCTOR ON LLM END")
        text = response.generations[0][0].text
        asyncio.run(self.websocket.send_json({"doctor": text}))


@app.on_event("startup")
async def startup_event() -> None:
    global kb
    global chain_chainer
    global symptom_name_to_symptom
    global matcher_chain
    global action_picker
    global chiefcomplaint_chain

    embedding_model_name = "bert-base-uncased"
    reader = kb_reading.CSVKnowledgeBaseReader("data/ClaudeKnowledgeBase.csv")
    kb = reader.load_knowledge_base()
    retrieval_config = retriever.HuggingFaceEncoderEmbeddingsConfig(
        model_name_or_path=embedding_model_name,
        device="cpu",
    )
    symptom_name_to_symptom = {}
    for symptom_group in kb.condition_symptoms.values():
        for weighted_symptom in symptom_group:
            symptom = datamodels.SymptomTransformer.to_symptom(weighted_symptom)
            symptom_name_to_symptom[symptom.name.strip()] = symptom

    texts = list(set(symptom_name_to_symptom))
    vectorstore = HuggingFAISS.from_model_config_and_texts(texts, retrieval_config)
    matcher_chain = matcher.MatchingChain.from_anthropic_and_retriever(
        symptom_extract_prompt=prompts.SYMPTOM_EXTRACT_PROMPT,
        symptom_match_prompt=prompts.SYMPTOM_MATCH_PROMPT,
        vec_store=vectorstore,
        parser=matcher.parse_raw_extract,
    )
    chiefcomplaint_chain = matcher.MatchingChain.from_anthropic_and_retriever(
        symptom_extract_prompt=cc_prompts.CC_EXTRACT_PROMPT,
        symptom_match_prompt=cc_prompts.CC_MATCH_PROMPT,
        vec_store=vectorstore,
        parser=matcher.parse_raw_extract_cc,
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
            diagnosis = await run_chain(
                websocket=websocket,
                note=message["content"],
                chainer=chainer,
            )
            await websocket.send_json({"condition": diagnosis.name})
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


K = 5


async def run_chain(
    websocket: WebSocket, note: str, chainer: chaining_the_chains.ChainChainer
) -> Optional[datamodels.Condition]:
    matrix = datamodels.DiseaseSymptomKnowledgeBaseTransformer.to_numpy(kb)
    state = action_states.SimulationNextActionState(matrix, discount_rate=1e-9)
    try:
        cc_matches_raw = chiefcomplaint_chain({"note": note})
        cc_matches: List[datamodels.SymptomMatch] = [
            datamodels.SymptomMatch(**match) for match in cc_matches_raw
        ]
        chief_complaints = [
            symptom_name_to_symptom[m.symptom_match.strip()] for m in cc_matches
        ]
    except Exception as e:
        logger.error(e)
        return
    logger.info("Chief complaints are {}", chief_complaints)
    await websocket.send_json(
        {"patient": f'I have {", ".join([c.name for c in chief_complaints])}'}
    )
    state.pertinent_pos.update(chief_complaints)
    rollout_policy = action_states.ArgMaxDiagnosisRolloutPolicy()
    searcher = multi_choice_mcts.MultiChoiceMCTS(
        timeLimit=3000, rolloutPolicy=rollout_policy
    )
    q_counter = 0
    while not isinstance(
        (actions := searcher.search(initialState=state, top_k=5))[0],
        datamodels.Condition,
    ):
        top_k = get_top_k_condition_probas(state, k=K)
        await websocket.send_json(
            {"brain": "\n".join([f"{c.name}: {p*100:.2f}%" for c, p in top_k])}
        )
        if max(top_k, key=lambda x: x[1])[1] > 0.8:
            await websocket.send_json(
                {"condition": f"Is it {max(top_k, key=lambda x: x[1])[0].name}?"}
            )
            return max(top_k, key=lambda x: x[1])[0]

        q_counter += 1
        if q_counter > 10:
            await websocket.send_json({"condition": "I'm sorry, I'm not sure."})
            await websocket.close()
            return
        assert isinstance(actions[0], datamodels.Symptom)
        logger.info(f"{actions=}")

        action_name = action_picker(actions=actions, state=state)

        patient_symptom_response = chainer.interaction(note, action_name)
        new_positives, new_negatives = make_new_symptoms(
            action_name, patient_symptom_response
        )

        logger.info(f"{new_positives=}")
        logger.info(f"{new_negatives=}")
        state.pertinent_pos.update(new_positives)
        state.pertinent_neg.update(new_negatives)

    diagnosis = actions[0]
    logger.info(f"Diagnosis: {diagnosis}")
    return diagnosis


def get_top_k_condition_probas(
    state: action_states.SimulationNextActionState, k: int
) -> List[Tuple[datamodels.Condition, float]]:
    condition_probas = state.getConditionProbabilityDict()
    top_k = sorted(
        condition_probas.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:K]
    logger.info(f"{top_k=}")
    return top_k


def make_new_symptoms(
    action_name: str, patient_symptom_response: List[datamodels.SymptomMatch]
):
    non_key_error_symptoms = []
    for s in patient_symptom_response:
        try:
            symptom_name_to_symptom[s.symptom_match.strip()]
            non_key_error_symptoms.append(s)
        except KeyError:
            pass

    if non_key_error_symptoms:
        new_positives = [
            symptom_name_to_symptom[s.symptom_match.strip()]
            for s in non_key_error_symptoms
            if s.present
        ]
        new_negatives = [
            symptom_name_to_symptom[s.symptom_match.strip()]
            for s in non_key_error_symptoms
            if not s.present
        ]
    else:
        new_positives = []
        new_negatives = [symptom_name_to_symptom[action_name.strip()]]
    return new_positives, new_negatives


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    ...
