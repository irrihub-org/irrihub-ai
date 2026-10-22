"""
Carrying the conversation to the CRM.

The Chatwoot conversation auto-resolves ten minutes after the handoff,
so a rep opening the lead usually cannot go back and read it. The
transcript travels with the lead instead.

It lands in a JSON column on every lead, so it is capped — and the tail
is kept rather than the head, because the recent exchange is what the
rep acts on and the opening of a long chat is usually greetings.
"""

from app.transcript import MAX_TRANSCRIPT_CHARS, trim_transcript


def test_a_short_conversation_is_kept_whole():
    text = "[USER] : hi\n[ASSISTANT] : Hello!\n[USER] : I need a quote"
    assert trim_transcript(text) == text


def test_a_long_conversation_is_capped():
    text = "x" * (MAX_TRANSCRIPT_CHARS * 3)
    trimmed = trim_transcript(text)

    assert len(trimmed) <= MAX_TRANSCRIPT_CHARS + 40  # + the notice


def test_trimming_keeps_the_end_not_the_beginning():
    """The rep acts on the recent exchange. The start of a long chat is
    usually greetings and general questions."""
    text = "OLDEST\n" + ("filler line\n" * 2000) + "NEWEST: my number is 0700123987"
    trimmed = trim_transcript(text)

    assert "NEWEST: my number is 0700123987" in trimmed
    assert "OLDEST" not in trimmed


def test_a_trimmed_transcript_says_so():
    """Otherwise it reads as if the conversation began mid-sentence."""
    text = "filler line\n" * 2000
    assert "earlier messages not shown" in trim_transcript(text)


def test_trimming_starts_at_a_message_boundary():
    """Cutting at a fixed offset would open the transcript halfway
    through somebody's sentence."""
    # Each line is numbered, so a partial one is detectable — a fixture
    # of identical characters would pass even when cut mid-line.
    text = "".join(f"[USER] : message number {i:05d}\n" for i in range(1000))
    trimmed = trim_transcript(text)

    body = trimmed.split("\n", 1)[1]
    assert body.startswith("[USER] :"), f"opened mid-line: {body[:40]!r}"
