"""
Tests for contact-detail capture, written against conversation 50 — the
lead that was lost because her email, phone and name were each treated as
a question and answered from the knowledge base.
"""

from app.contact_capture import acknowledgement, extract_contact_details

ASKED = "[ASSISTANT] : Give the team a way to reach you."


def test_conversation_50_email_is_captured():
    assert extract_contact_details("Ndutalm@gmail.com", ASKED) == {
        "email": "Ndutalm@gmail.com"
    }


def test_conversation_50_phone_is_captured_and_normalised():
    assert extract_contact_details("0721716571", ASKED) == {
        "phone": "+254721716571"
    }


def test_conversation_50_bare_name_is_captured():
    assert extract_contact_details("Lydia", ASKED) == {"full_name": "Lydia"}


def test_phone_formats_all_normalise_to_e164():
    for raw in [
        "0721716571",
        "+254721716571",
        "254721716571",
        "0721 716 571",
        "0721-716-571",
        "my number is 0721716571 please call",
    ]:
        got = extract_contact_details(raw, ASKED)
        assert got["phone"] == "+254721716571", raw


def test_a_longer_number_is_not_mistaken_for_a_phone():
    """An invoice or order number must not be captured as a phone."""
    assert extract_contact_details("order 07217165719999", ASKED) == {}


def test_safaricom_and_airtel_prefixes():
    got = extract_contact_details("0110123456", ASKED)
    assert got["phone"] == "+254110123456"
    got = extract_contact_details("0788112233", ASKED)
    assert got["phone"] == "+254788112233"


def test_phone_and_email_in_one_message():
    got = extract_contact_details("0721716571 / lydia@example.com", ASKED)
    assert got == {"phone": "+254721716571", "email": "lydia@example.com"}


def test_a_real_question_is_not_treated_as_contact_details():
    """The regression that matters: normal questions must still reach RAG."""
    for q in [
        "How much is a drip kit?",
        "What is the price of shade net?",
        "Do you deliver to Kisumu?",
        "I already have an irrigation plan and need a quote",
        "tomatoes",
        "maize",
    ]:
        assert extract_contact_details(q, ASKED) == {}, q


def test_a_name_is_only_captured_when_the_bot_asked():
    # Same word, no preceding request — must fall through to RAG.
    assert extract_contact_details("Lydia", None) == {}
    assert extract_contact_details("Lydia", "[USER] : how much is drip") == {}


def test_a_phone_is_captured_even_without_being_asked():
    """Unlike a name, a phone number is unambiguous on its own."""
    assert extract_contact_details("call me on 0721716571", None) == {
        "phone": "+254721716571"
    }


def test_self_introduction_is_captured_without_being_asked():
    """An explicit introduction is unambiguous, unlike a bare name."""
    for said, expected in [
        ("My name is Lydia", "Lydia"),
        ("my name is lydia wanjiku", "Lydia Wanjiku"),
        ("I am Lydia", "Lydia"),
        ("I'm Lydia", "Lydia"),
        ("Hi, this is Lydia", "Lydia"),
        ("naitwa Lydia", "Lydia"),
    ]:
        assert extract_contact_details(said, None) == {
            "full_name": expected
        }, said


def test_name_and_phone_in_one_message():
    """'I'm Lydia, call me on 0721716571' must not lose the name."""
    got = extract_contact_details("I'm Lydia, call me on 0721716571", None)
    assert got["full_name"] == "Lydia"
    assert got["phone"] == "+254721716571"


def test_the_new_handoff_wording_primes_name_capture():
    """The escalation message must match a marker, or the reply to it
    goes to the knowledge base instead of being read as a name."""
    handoff = (
        "[ASSISTANT] : I'm connecting you with one of our team members — "
        "someone will be with you shortly. May I have your name so they "
        "know who they're speaking with?"
    )
    assert extract_contact_details("Lydia", handoff) == {"full_name": "Lydia"}


def test_swahili_handoff_wording_primes_name_capture():
    handoff = "[ASSISTANT] : Naomba jina lako ili wajue wanazungumza na nani?"
    assert extract_contact_details("Wanjiku", handoff) == {
        "full_name": "Wanjiku"
    }


def test_an_introduction_of_a_non_name_is_rejected():
    assert extract_contact_details("I am ok", None) == {}
    assert extract_contact_details("this is maize", None) == {}


def test_short_pleasantries_are_not_names():
    for word in ["yes", "no", "ok", "thanks", "hello", "asante", "sawa"]:
        assert extract_contact_details(word, ASKED) == {}, word


def test_a_short_question_is_not_a_name():
    assert extract_contact_details("Price?", ASKED) == {}


def test_a_long_sentence_is_not_a_name():
    assert extract_contact_details(
        "I would like to know more about your products", ASKED
    ) == {}


def test_two_word_name_is_captured_and_title_cased():
    assert extract_contact_details("lydia wanjiku", ASKED) == {
        "full_name": "Lydia Wanjiku"
    }


def test_empty_message_is_safe():
    assert extract_contact_details("", ASKED) == {}
    assert extract_contact_details("   ", ASKED) == {}


def test_acknowledgement_repeats_what_was_captured():
    text = acknowledgement({"phone": "+254721716571"})
    assert "+254721716571" in text
    assert "team" in text.lower()


def test_acknowledgement_greets_by_name():
    assert "Lydia" in acknowledgement({"full_name": "Lydia"})


def test_acknowledgement_has_a_swahili_form():
    assert "Asante" in acknowledgement({"phone": "+254721716571"}, "sw")
