"""
LLM fallback.

Used only when Tier 1's keyword classifier (intent.py) returns an
empty category - currently only happens for general_query. Mirrors
the same two-tier pattern already used for escalation detection in
main.py (wants_human_by_keyword -> wants_human_by_intent): cheap
keyword check first, LLM call only as a fallback for what keywords
miss.

Reuses app.llm.generate_response so this automatically gets the same
timeout handling and OPENROUTER_CLASSIFIER_MODELS fallback list as
the escalation classifier, instead of duplicating that logic here.
"""

from app.llm import generate_response, OPENROUTER_CLASSIFIER_MODELS

FALLBACK_CATEGORY = "general_information"

CATEGORY_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a single farmer support question into ONE short "
    "category label (2-4 words, lowercase, underscores instead of "
    "spaces - e.g. 'irrigation_setup', 'crop_watering_schedule', "
    "'system_maintenance'). Reply with ONLY the category label, "
    "nothing else - no explanation, no punctuation."
)


async def classify_category_llm(message: str) -> str:
    """
    Ask a cheap LLM call to assign a short category label. Falls back
    to FALLBACK_CATEGORY if the call fails or returns something
    unusable, so this NEVER raises and NEVER returns an empty string.
    """
    try:
        raw_category = await generate_response(
            message,
            system_prompt=CATEGORY_CLASSIFIER_SYSTEM_PROMPT,
            models=OPENROUTER_CLASSIFIER_MODELS,
        )
    except Exception as e:
        print(f"Category classification failed ({type(e).__name__}), using fallback: {e}")
        return FALLBACK_CATEGORY

    cleaned = "".join(
        c for c in raw_category.strip().lower() if c.isalnum() or c == "_"
    )

    return cleaned[:50] if cleaned else FALLBACK_CATEGORY