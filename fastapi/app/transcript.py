"""
The conversation, trimmed to fit on a lead.

A rep opening a lead usually cannot go back and read the chat: Chatwoot
auto-resolves an escalated conversation ten minutes after the handoff,
and that stays until WhatsApp lands. So the conversation travels with
the lead instead of being left where nobody will look for it.

Kept out of main so it can be tested without importing FastAPI and the
embedding model with it.
"""

# This lands in a JSON column on every lead, so it is capped.
MAX_TRANSCRIPT_CHARS = 8000

TRIMMED_NOTICE = "[earlier messages not shown]"

# How far into the kept text to look for a line break before giving up
# and starting mid-line.
_BOUNDARY_SEARCH = 200


def trim_transcript(text: str) -> str:
    """Cap a conversation, keeping the end.

    The tail is kept rather than the head: the recent exchange is what
    the rep acts on, and the opening of a long chat is usually
    greetings and general questions.
    """
    if len(text) <= MAX_TRANSCRIPT_CHARS:
        return text

    kept = text[-MAX_TRANSCRIPT_CHARS:]

    # Start at a message boundary, so the transcript does not open
    # halfway through somebody's sentence.
    newline = kept.find("\n")
    if 0 <= newline < _BOUNDARY_SEARCH:
        kept = kept[newline + 1:]

    return f"{TRIMMED_NOTICE}\n{kept}"
