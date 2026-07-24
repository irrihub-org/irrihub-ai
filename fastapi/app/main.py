from contextlib import asynccontextmanager

from fastapi import FastAPI
from langdetect import detect_langs, DetectorFactory, LangDetectException

# langdetect uses a random seed internally by default, which makes
# results non-deterministic across runs. Seed it for consistency.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "sw"}
from sentence_transformers import SentenceTransformer

from app.rag import retrieve_top_k
from app.llm import generate_response, OPENROUTER_CLASSIFIER_MODELS
from app.models import ChatRequest, AIResponse
from app.db import is_conversation_human_managed, mark_conversation_human_managed

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Don't load the model during startup
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
        "message": "IrriHub AI — endpoint not configured yet"
    }


def detect_language(text: str) -> str:
    """
    Detect the language of the user's message, constrained to the
    languages this system actually supports (en, sw). Falls back to
    'en' if detection fails entirely (e.g. message too short/empty).

    langdetect recognises 55+ languages, but on short or ambiguous
    text it can confidently return something completely unrelated
    (e.g. 'it', 'tl') that this system has no business acting on.
    Rather than trust its raw top guess, we look at ALL of its
    candidate guesses and pick whichever of our two supported
    languages scored highest. If neither en nor sw appears in its
    candidates at all, we default to 'en'.

    NOTE: langdetect is not trained on Sheng (Kenyan slang), so Sheng
    messages will likely be detected as 'sw' or occasionally 'en'.
    Sheng normalisation (build order Step 4) should run BEFORE this
    once normalise.py exists, since normalising Sheng into standard
    Swahili first will make detection more reliable.
    """
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "en"

    supported_candidates = [c for c in candidates if c.lang in SUPPORTED_LANGUAGES]

    if not supported_candidates:
        return "en"

    return supported_candidates[0].lang


LANGUAGE_NAMES = {
    "en": "English",
    "sw": "Swahili",
}

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

# Once a conversation is escalated, a human owns it — the bot should
# stay quiet rather than keep answering underneath them. This is the
# message returned (with route="escalate") for any further /chat call
# on a conversation already marked human-managed.
ALREADY_ESCALATED_MESSAGES = {
    "en": "A team member is already helping with this conversation.",
    "sw": "Mfanyakazi wetu tayari anakusaidia katika mazungumzo haya.",
}


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
        print(f"Intent classification failed, defaulting to CONTINUE: {e}")
        return False

    return verdict.strip().upper().startswith("ESCALATE")


@app.post("/chat", response_model=AIResponse)
async def chat(req: ChatRequest):
    query = req.message
    language = detect_language(query)

    if await is_conversation_human_managed(req.conversation_id):
        return AIResponse(
            text=ALREADY_ESCALATED_MESSAGES.get(
                language, ALREADY_ESCALATED_MESSAGES["en"]
            ),
            route="escalate",
            language=language,
        )

    if wants_human_by_keyword(query) or await wants_human_by_intent(query):
        await mark_conversation_human_managed(req.conversation_id, language)
        return AIResponse(
            text=ESCALATION_MESSAGES.get(language, ESCALATION_MESSAGES["en"]),
            route="escalate",
            language=language,
        )

    # Load the model only on the first request
    if app.state.embedding_model is None:
        print("Loading embedding model...")
        app.state.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Embedding model loaded.")

    model = app.state.embedding_model
    language_name = LANGUAGE_NAMES.get(language, language)

    chunks = await retrieve_top_k(query, model)

    context_text = "\n".join(
        chunk["chunk_text"] for chunk in chunks
    )

    history_block = ""
    if req.conversation_history:
        history_block = f"""
Conversation so far:
{req.conversation_history}
"""

    prompt = f"""
You are an agricultural assistant for IrriHub, serving farmers in Kenya.

Use the information provided in the context below to answer the user's question.
- If the context fully answers the question, answer directly and confidently. Do not add any disclaimer about missing information.
- If the context only partially answers the question, give the best answer you can from what's there, then briefly note which part you couldn't address.
- Only if the context has nothing relevant to the question at all, say that you do not have enough information — do not say this after already giving a substantive answer.
- The context may include reference material written for other countries or regions. If a detail in the context is tied to a specific country/region other than Kenya (e.g. local regulations, region-specific product availability), do not state it as if it applies locally — either omit it or clearly note it's from general/regional reference material rather than local guidance.
Respond in {language_name} ({language}).

Context:
{context_text}
{history_block}
User question:
{query}
"""

    answer = await generate_response(prompt)

    return AIResponse(
        text=answer,
        language=language,
        sources=[str(chunk["id"]) for chunk in chunks],
    )