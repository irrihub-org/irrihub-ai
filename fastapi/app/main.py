from contextlib import asynccontextmanager

from fastapi import FastAPI
from langdetect import detect_langs, DetectorFactory, LangDetectException
from sentence_transformers import SentenceTransformer

from app.rag import retrieve_top_k
from app.llm import generate_response
from app.models import ChatRequest, AIResponse
from app.intent import classify_intent
from app.escalation import determine_route, log_kb_gap

DetectorFactory.seed = 0
SUPPORTED_LANGUAGES = {"en", "sw"}

LANGUAGE_NAMES = {
    "en": "English",
    "sw": "Swahili",
}

HARDWARE_FAULT_DIAGNOSTIC = (
    "Pole sana kwa tatizo hili! Ili tuweze kukusaidia haraka: "
    "1) Je, pump inajaribu kuwasha lakini haina nguvu, au haianzishi kabisa? "
    "2) Angalia voltage kwenye solar controller - inapaswa kuwa 24V au zaidi. "
    "3) Angalia kama waya zote zimeunganishwa vizuri."
)

ESCALATION_MESSAGE = (
    "I'm not fully confident I have the right information for this. "
    "Let me connect you with our support team who can help further."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedding_model = None
    yield


app = FastAPI(
    title="IrriHub AI",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "irrihub-fastapi",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    return {
        "message": "IrriHub AI — Welcome"
    }


def detect_language(text: str) -> str:
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "en"

    supported_candidates = [c for c in candidates if c.lang in SUPPORTED_LANGUAGES]

    if not supported_candidates:
        return "en"

    return supported_candidates[0].lang


@app.post("/chat", response_model=AIResponse)
async def chat(req: ChatRequest):
    if app.state.embedding_model is None:
        print("Loading embedding model...")
        app.state.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Embedding model loaded.")

    model = app.state.embedding_model
    query = req.message

    language = detect_language(query)
    language_name = LANGUAGE_NAMES.get(language, language)

    intent, category_hint = classify_intent(query)

    if intent == "hardware_fault":
        return AIResponse(
            text=HARDWARE_FAULT_DIAGNOSTIC,
            intent=intent,
            category="pump_troubleshooting",
            confidence=1.0,
            route="deterministic",
            language=language,
            sources=[],
        )

    chunks = await retrieve_top_k(query, model)

    top_distance = chunks[0]["distance"] if chunks else None
    confidence, route, needs_caveat = determine_route(top_distance)

    print(
        f"[confidence-gate] query={query!r} "
        f"top_distance={top_distance} confidence={confidence:.4f} "
        f"route={route} needs_caveat={needs_caveat}"
    )

    if route == "escalate":
        failure_reason = "no_kb_match" if top_distance is None or top_distance > 0.6 else "low_confidence"
        await log_kb_gap(
            query_text=query,
            confidence=confidence,
            channel=req.channel,
            failure_reason=failure_reason,
        )
        top_category = chunks[0]["category"] if chunks else ""
        return AIResponse(
            text=ESCALATION_MESSAGE,
            intent=intent,
            category=top_category,
            confidence=confidence,
            route="escalate",
            language=language,
            sources=[str(chunk["id"]) for chunk in chunks],
        )

  
    PER_CHUNK_DISTANCE_CUTOFF = 1.0
    relevant_chunks = [
        chunk for chunk in chunks
        if chunk["distance"] is None or chunk["distance"] <= PER_CHUNK_DISTANCE_CUTOFF
    ] or chunks[:1]

    context_text = "\n".join(chunk["chunk_text"] for chunk in relevant_chunks)

    history_block = ""
    if req.conversation_history:
        history_block = f"""
Conversation so far:
{req.conversation_history}
"""

    prompt = f"""
You are an agricultural assistant for IrriHub, a company selling irrigation
equipment in Kenya.

STRICT RULES:
1. Answer ONLY using facts explicitly stated in the Context below.
2. Do NOT describe yourself, your role, your capabilities, or your
   limitations. Never say things like "I am an assistant" or "I don't
   have expertise in X" - these are not in the Context and must not
   be invented.
3. If the Context does not contain information that directly answers
   the question, respond with EXACTLY this sentence and nothing else:
   "I don't have that information — let me connect you with our support team."
4. Do not guess, infer, or fill gaps with general knowledge about
   irrigation or agriculture that isn't explicitly in the Context.
5. Respond in {language_name} ({language}).

Context:
{context_text}
{history_block}
User question:
{query}
"""

    answer = await generate_response(prompt)

    # Medium-confidence band: respond, but add a caveat + escalation offer.
    if needs_caveat:
        answer += (
            "\n\n(I found some relevant information, but I'm not fully "
            "certain this fully answers your question — let me know if "
            "you'd like to speak with an agronomist.)"
        )

    top_category = relevant_chunks[0]["category"] if relevant_chunks else ""

    return AIResponse(
        text=answer,
        intent=intent,
        category=top_category,
        confidence=confidence,
        route=route,
        language=language,
        sources=[str(chunk["id"]) for chunk in relevant_chunks],
    )