"""
Which messages are a genuine request for a human.

Kept out of main so it can be tested without importing FastAPI and the
embedding model with it — same reason handoff.py lives apart.

Conversation 15 is why the guard exists. The visitor answered "yes" to
"are you a farmer in Kenya?", the LLM classifier saw that single word
with none of the conversation around it, guessed ESCALATE, and the
conversation was handed to a human before it had started.
"""

# Replies that answer a question rather than make a request. Sent to the
# classifier on their own they are ambiguous, and it guesses. These fall
# through to the knowledge base instead, where the conversation history
# gives the model something to work with.
NOT_A_REQUEST = {
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "no", "nope",
    "hi", "hello", "hey", "thanks", "thank you", "cheers",
    "ndio", "ndiyo", "hapana", "sawa", "asante", "habari",
}


def is_too_short_to_judge(text: str) -> bool:
    """Whether this message should skip the escalation classifier."""
    return text.strip().lower().rstrip("!.?") in NOT_A_REQUEST
