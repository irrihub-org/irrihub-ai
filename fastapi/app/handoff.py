"""
The escalation handoff message.

Kept out of main so it can be tested without importing the FastAPI app
and the embedding model with it — this is copy, not machinery.
"""

# The handoff. Three jobs, in order of how much they cost us when
# missing:
#
#  1. Ask for whatever contact detail we do not already have. A lead with
#     no phone cannot be called; a lead with no name reaches the rep as
#     "Chatwoot Visitor". We ask for the missing one rather than both, so
#     it reads as a sentence and not a form.
#  2. Say the team will CALL BACK, not that someone is arriving in chat.
#     Chatwoot auto-resolves the conversation ten minutes after the
#     handoff (kept until WhatsApp lands), so "someone will be with you
#     shortly" was a promise the channel could not keep.
#  3. Say the chat will close, so the silence that follows reads as
#     expected rather than as being ignored.
#
# The name-ask wording matches a marker in contact_capture, so a one-word
# reply is recognised as a name instead of being sent to the knowledge
# base.
_HANDOFF = {
    "en": {
        "open": "I'm passing this to our team — they'll call you back today.",
        "ask_phone": " What number should they call you on?",
        "ask_name": " May I have your name so they know who they're speaking with?",
        "ask_both": " May I have your name and the best number to call you on?",
        "close": " I'll close this chat now, but your request is with them.",
    },
    "sw": {
        "open": "Ninawasilisha hili kwa timu yetu — watakupigia simu leo.",
        "ask_phone": " Ni nambari gani wakupigie?",
        "ask_name": " Naomba jina lako ili wajue wanazungumza na nani?",
        "ask_both": " Naomba jina lako na nambari ya simu wakupigie?",
        "close": " Nitafunga mazungumzo haya sasa, lakini ombi lako liko kwao.",
    },
}


def build_handoff_message(
    language: str,
    has_name: bool = False,
    has_phone: bool = False,
    lead_in: str = "",
) -> str:
    """The escalation message, asking only for what is missing.

    lead_in prefixes it for the low-confidence path, which has to explain
    why it is handing over before it does so.
    """
    copy = _HANDOFF.get(language, _HANDOFF["en"])

    if not has_phone and not has_name:
        ask = copy["ask_both"]
    elif not has_phone:
        ask = copy["ask_phone"]
    elif not has_name:
        ask = copy["ask_name"]
    else:
        ask = ""

    # The closing line only goes out when we are NOT asking for anything.
    # "I'll close this chat now" directly after a question contradicts
    # itself — it invites a reply and refuses it in the same breath. When
    # we have asked, the visitor's answer is the next message, and the
    # acknowledgement in contact_capture is where the conversation ends.
    tail = copy["close"] if not ask else ""

    return f"{lead_in}{copy['open']}{ask}{tail}"


# Prefixes the handoff when the bot is escalating because the knowledge
# base had nothing, rather than because the user asked for a human.
LOW_CONFIDENCE_LEAD_IN = {
    "en": "I'm not fully confident I have the right information for this. ",
    "sw": "Sina uhakika kamili kuwa nina jibu sahihi kwa hili. ",
}

