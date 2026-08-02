from contextlib import asynccontextmanager
import re

from fastapi import FastAPI
from langdetect import detect_langs, DetectorFactory, LangDetectException
from sentence_transformers import SentenceTransformer

from app.rag import retrieve_top_k
from app.llm import generate_response, OPENROUTER_CLASSIFIER_MODELS
from app.models import ChatRequest, AIResponse
from app.db import is_conversation_human_managed, mark_conversation_human_managed
from app.intent import classify_intent
from app.escalation import determine_route, log_kb_gap
from app.category_classifier import classify_category_llm

DetectorFactory.seed = 0
SUPPORTED_LANGUAGES = {"en", "sw"}

LANGUAGE_NAMES = {
    "en": "English",
    "sw": "Swahili",
}

INTENT_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a single user message for a customer support chatbot. "
    "Reply with exactly one word: ESCALATE or CONTINUE.\n"
    "ESCALATE means the user is asking to speak with a human, an agent, "
    "an expert, customer support/care, or otherwise wants to stop talking "
    "to the bot and be connected to a person.\n"
    "CONTINUE means the user is asking a normal question the bot should "
    "try to answer itself.\n"
    "Reply with only ESCALATE or CONTINUE, nothing else."
)

# Phrases that mean the user wants a human, regardless of what the KB
# contains. Checked before RAG/LLM so escalation never depends on the
# model correctly interpreting the request from retrieved context.
ESCALATION_PHRASES = {
    "escalate to human",
    "talk to a human",
    "speak to a human",
    "talk to an agent",
    "speak to an agent",
    "connect me to a human",
    "connect me to an agent",
    "human agent",
    "real person",
    "customer care",
    "customer service",
}

ESCALATION_MESSAGES = {
    "en": "I'm connecting you with one of our team members — someone will be with you shortly.",
    "sw": "Ninakuunganisha na mmoja wa wafanyakazi wetu — mtu atakusaidia hivi karibuni.",
}

# Once a conversation is escalated, a human owns it - the bot should
# stay quiet rather than keep answering underneath them.
ALREADY_ESCALATED_MESSAGES = {
    "en": "A team member is already helping with this conversation.",
    "sw": "Mfanyakazi wetu tayari anakusaidia katika mazungumzo haya.",
}

# Distinct from the explicit human-request escalation above - this
# fires when the user DIDN'T ask for a human, but the KB genuinely has
# nothing relevant (confidence gate, see escalation.py).
LOW_CONFIDENCE_MESSAGE = (
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


def wants_human_by_keyword(text: str) -> bool:
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in ESCALATION_PHRASES)


async def wants_human_by_intent(text: str) -> bool:
    """
    Keyword matching only catches phrasings we thought to list. Real
    users ask in ways we won't anticipate (e.g. "do you have an expert
    that can assist?"), so anything the keyword check doesn't already
    catch gets a cheap LLM classification pass before falling through
    to the full RAG pipeline.
    """
    try:
        verdict = await generate_response(
            text,
            system_prompt=INTENT_CLASSIFIER_SYSTEM_PROMPT,
            models=OPENROUTER_CLASSIFIER_MODELS,
        )
    except Exception as e:
        print(f"Intent classification failed ({type(e).__name__}), defaulting to CONTINUE: {e}")
        return False

    return verdict.strip().upper().startswith("ESCALATE")


def _extract_chunk_label(chunk_text: str) -> str:
    """
    Derive a short, human-readable label describing what a chunk is
    about, from its own content. Three patterns, checked in order:
    1. FAQ-style chunks ("Q: ...\\nA: ...") -> use the question itself.
    2. ALL-CAPS section headers (e.g. "TROUBLESHOOTING") -> use that.
    3. Fallback -> a short truncated snippet of the first line.
    """
    first_line = chunk_text.strip().split("\n", 1)[0].strip()

    if first_line.startswith("Q:"):
        question = first_line[2:].strip()
        return question if len(question) <= 80 else question[:77] + "..."

    if re.fullmatch(r"[A-Z][A-Z\s]{2,}", first_line):
        return first_line.title()

    return first_line if len(first_line) <= 60 else first_line[:57] + "..."


def _build_source_labels(chunks: list[dict]) -> list[str]:
    """
    Returns a deduplicated, order-preserving list of "filename —
    label" strings, so the response's sources field is more specific
    than a bare filename or a raw chunk UUID.
    """
    seen = set()
    labels = []
    for chunk in chunks:
        filename = chunk.get("source_filename")
        if not filename:
            continue
        label = _extract_chunk_label(chunk.get("chunk_text", ""))
        entry = f"{filename} — {label}" if label else filename
        if entry not in seen:
            seen.add(entry)
            labels.append(entry)
    return labels


@app.post("/chat", response_model=AIResponse)
async def chat(req: ChatRequest):
    query = req.message
    language = detect_language(query)

    # --- Tier 0: is this conversation already handed to a human? ---
    if await is_conversation_human_managed(req.conversation_id):
        return AIResponse(
            text=ALREADY_ESCALATED_MESSAGES.get(
                language, ALREADY_ESCALATED_MESSAGES["en"]
            ),
            route="escalate",
            language=language,
        )

    # --- Tier 1: did the user explicitly ask for a human? ---
    if wants_human_by_keyword(query) or await wants_human_by_intent(query):
        await mark_conversation_human_managed(req.conversation_id, language)
        return AIResponse(
            text=ESCALATION_MESSAGES.get(language, ESCALATION_MESSAGES["en"]),
            intent="escalation_request",
            category="human_handoff_requested",
            route="escalate",
            language=language,
        )

    if app.state.embedding_model is None:
        print("Loading embedding model...")
        app.state.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Embedding model loaded.")

    model = app.state.embedding_model
    language_name = LANGUAGE_NAMES.get(language, language)

    # category here comes from the INTENT classifier (a sub-type
    # within the intent, e.g. "pricing" within product_inquiry).
    intent, category = classify_intent(query)

    # Tier 2: keyword classification didn't find a category (only
    # happens for general_query) - fall back to a cheap LLM call so
    # category is never blank. Mirrors the escalation classifier's
    # keyword-first, LLM-fallback pattern above.
    if not category:
        category = await classify_category_llm(query)

    

    chunks = await retrieve_top_k(query, model)

    top_distance = chunks[0]["distance"] if chunks else None
    confidence, route, needs_caveat = determine_route(top_distance)

    print(
        f"[confidence-gate] query={query!r} "
        f"top_distance={top_distance} confidence={confidence:.4f} "
        f"route={route} needs_caveat={needs_caveat}"
    )

    # Tier 2: KB match too weak, even though user didn't ask for
    # a human explicitly. Distinct from Tier 1 above. 
    if route == "escalate":
        failure_reason = (
            "no_kb_match" if top_distance is None or top_distance > 0.6
            else "low_confidence"
        )
        await log_kb_gap(
            query_text=query,
            confidence=confidence,
            channel=req.channel,
            failure_reason=failure_reason,
        )
        return AIResponse(
            text=LOW_CONFIDENCE_MESSAGE,
            intent=intent,
            category=category,
            confidence=confidence,
            route="escalate",
            language=language,
            sources=_build_source_labels(chunks),
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
You are an agricultural assistant for IrriHub, serving farmers in Kenya.
You handle two different kinds of questions differently:

1. IRRIHUB-SPECIFIC QUESTIONS (products, pricing, availability, installation
   steps, troubleshooting, warranty, company policies): answer ONLY using
   the Context below. Never invent prices, specs, or steps that aren't
   explicitly there. If the Context has nothing relevant to this type of
   question, say you do not have enough information and offer to connect
   the user with support.

2. GENERAL AGRONOMY / FARMING KNOWLEDGE QUESTIONS (crop facts, growing
   conditions, general agricultural practices - not specific to IrriHub's
   products or services): you may answer using your own general knowledge,
   even if nothing relevant appears in the Context, and even if the
   question isn't about Kenya. Do not refuse or claim ignorance just
   because the Context doesn't cover it - the Context is only required
   for IrriHub-specific questions, not general agricultural knowledge.

Rules that apply to both:
- If the Context fully answers the question, answer directly and
  confidently. Do not add a disclaimer about missing information.
- If the Context only partially answers the question, give the best
  answer you can from what's there, then note what it didn't cover.
- The Context may include reference material written for other countries
  or regions. If a Context detail is tied to a specific country/region
  other than Kenya (e.g. local regulations, region-specific product
  availability), do not present it as if it applies locally in Kenya -
  either omit it or clearly note it's general/regional reference material.
- Respond in {language_name} ({language}).

Context:
{context_text}
{history_block}
User question:
{query}
"""

    answer = await generate_response(prompt)

    if needs_caveat:
        answer += (
            "\n\n(I found some relevant information, but I'm not fully "
            "certain this fully answers your question — let me know if "
            "you'd like to speak with an agronomist.)"
        )

    return AIResponse(
        text=answer,
        intent=intent,
        category=category,
        confidence=confidence,
        route=route,
        language=language,
        sources=_build_source_labels(relevant_chunks),
    )