"""
Tests for the handoff message.

Two things it has to get right, both learned from conversation 50:

  - Ask for what is MISSING. Asking a visitor for a name Chatwoot already
    has wastes the one message they are most likely to answer, and the
    detail that actually makes a lead callable is the phone number.
  - Promise a callback, not an imminent reply. Chatwoot auto-resolves the
    conversation ten minutes after the handoff, so "someone will be with
    you shortly" is a promise the channel cannot keep.
"""

from app.handoff import build_handoff_message


def test_asks_for_both_when_we_know_nothing():
    msg = build_handoff_message("en")
    assert "your name and the best number" in msg


def test_asks_only_for_the_phone_when_the_name_is_known():
    msg = build_handoff_message("en", has_name=True)
    assert "What number should they call you on?" in msg
    assert "your name" not in msg


def test_asks_only_for_the_name_when_the_phone_is_known():
    msg = build_handoff_message("en", has_phone=True)
    assert "May I have your name" in msg
    assert "number should they call" not in msg


def test_asks_for_nothing_when_both_are_known():
    msg = build_handoff_message("en", has_name=True, has_phone=True)
    assert "?" not in msg


def test_always_promises_a_callback_not_an_imminent_reply():
    """The channel closes in ten minutes; it cannot deliver 'shortly'."""
    for kwargs in [
        {},
        {"has_name": True},
        {"has_phone": True},
        {"has_name": True, "has_phone": True},
    ]:
        msg = build_handoff_message("en", **kwargs)
        assert "call you back today" in msg, kwargs
        assert "shortly" not in msg, kwargs


def test_says_the_chat_will_close_when_nothing_is_being_asked():
    """Otherwise the silence after auto-resolve reads as being ignored."""
    msg = build_handoff_message("en", has_name=True, has_phone=True)
    assert "close this chat" in msg


def test_does_not_announce_closing_while_asking_a_question():
    """'I'll close this chat' right after a question contradicts itself —
    it invites a reply and refuses it in the same breath. The closing
    line moves to the acknowledgement, once the visitor has answered."""
    for kwargs in [{}, {"has_name": True}, {"has_phone": True}]:
        msg = build_handoff_message("en", **kwargs)
        assert "?" in msg, kwargs
        assert "close this chat" not in msg, kwargs


def test_swahili_is_a_full_translation_not_a_fallback():
    msg = build_handoff_message("sw")
    assert "watakupigia simu leo" in msg
    assert "call you back" not in msg

    closing = build_handoff_message("sw", has_name=True, has_phone=True)
    assert "Nitafunga mazungumzo" in closing


def test_swahili_asks_only_for_what_is_missing():
    assert "Ni nambari gani" in build_handoff_message("sw", has_name=True)
    assert "Naomba jina lako" in build_handoff_message("sw", has_phone=True)


def test_an_unknown_language_falls_back_to_english():
    assert build_handoff_message("fr") == build_handoff_message("en")


def test_the_low_confidence_lead_in_prefixes_the_same_message():
    plain = build_handoff_message("en", has_name=True, has_phone=True)
    with_lead_in = build_handoff_message(
        "en", has_name=True, has_phone=True, lead_in="Not sure about that. "
    )
    assert with_lead_in == "Not sure about that. " + plain


def test_the_message_reads_as_one_sentence_run():
    """Spacing is built by concatenation, so it is easy to get wrong."""
    msg = build_handoff_message("en")
    assert "  " not in msg
    assert ".M" not in msg and ".I" not in msg
