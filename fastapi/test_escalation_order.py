"""
Which messages are a genuine request for a human.

Conversation 15 is why this exists. The visitor said "yes" — confirming
he was a farmer — the bot escalated, n8n assigned the conversation to
the Support team, and `Check If Already Assigned` then stopped the
workflow ever calling /chat again. His next message, "Chatika , number
is 0700123654", reached nothing. The lead landed as "Chatwoot Visitor"
with no phone and the farmer sat looking at silence.

Escalating is what silences the bot, so the bar for it has to be a real
request, not a guess at an ambiguous word.
"""

from app.escalation_intent import NOT_A_REQUEST, is_too_short_to_judge


def test_the_message_that_mis_escalated_conversation_15():
    assert is_too_short_to_judge("yes") is True


def test_short_acknowledgements_skip_the_classifier():
    """Asked to judge these without the question they answer, the
    classifier guesses — and guessed ESCALATE on "yes"."""
    for word in ["yes", "Yes!", "YES", "ok", "sure", "no", "thanks", " okay "]:
        assert is_too_short_to_judge(word) is True, word


def test_swahili_replies_are_covered_too():
    for word in ["ndio", "hapana", "sawa", "asante", "habari"]:
        assert is_too_short_to_judge(word) is True, word


def test_a_real_request_is_still_judged():
    """The guard must not swallow genuine requests phrased in ways the
    keyword list does not cover."""
    for phrase in [
        "do you have an expert who can help?",
        "I'd like to speak to an agent",
        "can someone call me",
        "yes please connect me to a person",
    ]:
        assert is_too_short_to_judge(phrase) is False, phrase


def test_a_question_mark_does_not_make_a_reply_a_request():
    """'ok?' is still an acknowledgement."""
    assert is_too_short_to_judge("ok?") is True


def test_the_guard_list_is_replies_not_requests():
    """A word here means 'this answers a question'. Anything that could
    be a request for help does not belong."""
    for word in ["agent", "human", "help", "support", "call"]:
        assert word not in NOT_A_REQUEST, word
