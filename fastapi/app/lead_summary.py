"""
A one-line summary of what the farmer actually wants, for the CRM lead.

Chatwoot auto-resolves an escalated conversation after 10 minutes of
inactivity. That stays in place until WhatsApp is integrated, which means
the conversation is NOT a reliable place for a rep to go and read what
the lead was about — by the time anybody looks, it is closed.

So the lead record has to stand on its own. Without this, a rep opening
Lydia's lead sees "Chatbot conversation escalated to a human agent" and
has to go digging in Chatwoot to discover she has an irrigation plan,
wants a quote, and asked to be called at 9am.

Kept separate from contact_capture because it answers a different
question — that module asks "who is this?", this one asks "what do they
want?" — and because this one makes an LLM call and must therefore be
allowed to fail without taking the escalation down with it.
"""

from app.llm import generate_response, OPENROUTER_CLASSIFIER_MODELS

_SYSTEM_PROMPT = (
    "You summarise a customer chat for a sales rep who will phone the "
    "customer back and has not read the conversation. Reply with ONE "
    "sentence, at most 25 words, in English. State what the customer "
    "wants and any detail that affects the callback — a requested time, "
    "a crop, an acreage, a location, a budget. Do not greet, do not "
    "explain, do not add pleasantries. If the conversation shows no "
    "clear request, reply exactly: No specific request stated."
)

MAX_SUMMARY_CHARS = 500


async def summarise_request(conversation_history: str | None) -> str | None:
    """One line a rep can act on, or None if there is nothing to say.

    Never raises: a summary is a convenience, and the escalation it is
    attached to matters more than it does.
    """
    if not conversation_history or not conversation_history.strip():
        return None

    # The tail carries the request; earlier turns are usually the bot
    # answering general agronomy questions.
    excerpt = conversation_history[-2000:]

    try:
        summary = await generate_response(
            f"Conversation:\n{excerpt}\n\nOne-sentence summary for the rep:",
            system_prompt=_SYSTEM_PROMPT,
            models=OPENROUTER_CLASSIFIER_MODELS,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        print(f"[lead-summary] failed, continuing without it: {exc}")
        return None

    if not summary:
        return None

    cleaned = " ".join(summary.split()).strip().strip('"')

    if not cleaned or cleaned.lower().startswith("no specific request"):
        return None

    return cleaned[:MAX_SUMMARY_CHARS]
