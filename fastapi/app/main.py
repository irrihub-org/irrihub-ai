from contextlib import asynccontextmanager

from fastapi import FastAPI
from langdetect import detect_langs, DetectorFactory, LangDetectException

# langdetect uses a random seed internally by default, which makes
# results non-deterministic across runs. Seed it for consistency.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "sw"}
from sentence_transformers import SentenceTransformer

from app.rag import retrieve_top_k
from app.llm import generate_response
from app.models import ChatRequest, AIResponse


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


@app.post("/chat", response_model=AIResponse)
async def chat(req: ChatRequest):
    # Load the model only on the first request
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
You are an agricultural assistant.

Use ONLY the information provided in the context below to answer the user's question.
If the answer cannot be found in the context, say that you do not have enough information.
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