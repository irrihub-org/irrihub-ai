"""
Contact-detail capture.

When the bot asks "how can we reach you?", the visitor's reply is a phone
number, an email address or a name — not a question. Those messages were
going through the same path as everything else: embedded, matched against
the knowledge base, and answered.

Conversation 50 is the worked example. The visitor gave her email, her
phone and her name across three messages and got, in order, "I don't have
enough information", an unprompted answer about drip-line pricing, and an
answer about raising seedlings in a shadehouse. A bare name like "Lydia"
is a short string with no semantic content, so pgvector returns whatever
is nearest and the model dutifully answers it.

So this runs BEFORE retrieval and short-circuits it. Detection is
deliberately regex-based rather than a model call: it must be cheap enough
to sit in front of every message, and an email address is not a judgement
call.

What is extracted ends up in AIResponse.crm_payload, which already existed
as an unused field on the model — n8n reads it to create the CRM lead.
"""

import re

# Kenyan mobiles reach the bot in every format people type them in:
# 0721716571, +254721716571, 254721716571, 0721 716 571. Separators are
# stripped before matching so one pattern covers them all.
_PHONE_RE = re.compile(r"(?<![\d])\+?(?:254|0)(?:7|1)\d{8}(?![\d])")

# Deliberately permissive — this identifies "that was an email address",
# it does not validate deliverability.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# A name is the hard case: "Lydia" and "maize" are both single words. We
# only treat a message as a name when the bot has just asked for one,
# which the caller establishes from conversation history.
_NAME_ASK_MARKERS = (
    "give the team a way to reach you",
    "a way to reach you",
    "your name",
    "may i have your name",
    "who am i speaking",
    "who they're speaking with",
    "how can we reach you",
    "provide me with your name",
    # Swahili handoff wording — see ESCALATION_MESSAGES in main.py.
    "jina lako",
    "wanazungumza na nani",
)

# "My name is Lydia", "I am Lydia", "Hi, this is Lydia" — an explicit
# self-introduction is unambiguous, so it is captured whether or not we
# asked first.
_NAME_INTRO_RE = re.compile(
    r"\b(?:my name is|i am|i'm|this is|naitwa|jina langu ni)\s+"
    r"([A-Za-z'’\-]+(?:\s+[A-Za-z'’\-]+){0,3})",
    re.IGNORECASE,
)

# Words that look like a name by shape but are plainly something else, so
# a one-word product question is never swallowed as a name.
_NOT_NAMES = {
    "yes", "no", "ok", "okay", "hi", "hello", "hey", "thanks", "thank you",
    "maize", "tomato", "tomatoes", "kale", "onion", "onions", "potato",
    "potatoes", "cabbage", "spinach", "drip", "pump", "greenhouse",
    "shadehouse", "price", "prices", "pricing", "quote", "help", "sure",
    "ndio", "hapana", "asante", "sawa", "habari", "karibu",
}

MAX_NAME_WORDS = 4

# Words people put around a number when answering "name and number?".
# "the number is 0700123987" has no name in it, and without this the
# lead-in words become one — a lead called "The Number Is".
_NAME_FILLER = {
    "the", "my", "name", "is", "number", "no", "phone", "mobile", "tel",
    "call", "me", "on", "and", "its", "it's", "am", "this", "here",
    "nambari", "yangu", "jina", "ni", "simu",
}


def _normalise_phone(raw: str) -> str:
    """To E.164, the format the CRM stores and dedups on."""
    digits = re.sub(r"\D", "", raw)

    if digits.startswith("254"):
        return "+" + digits
    if digits.startswith("0"):
        return "+254" + digits[1:]
    return "+254" + digits


def _bot_asked_for_contact(conversation_history: str | None) -> bool:
    """Did the assistant recently ask for a name or contact details?

    Only the tail of the history is considered: a request made twenty
    messages ago should not make an unrelated one-word reply look like a
    name.
    """
    if not conversation_history:
        return False

    tail = conversation_history[-600:].lower()
    return any(marker in tail for marker in _NAME_ASK_MARKERS)


def extract_contact_details(
    message: str,
    conversation_history: str | None = None,
) -> dict:
    """Pull any contact details out of a single message.

    Returns {} when the message carries none, which is the signal to
    carry on with normal retrieval.
    """
    found: dict = {}
    text = message.strip()

    if not text:
        return found

    email_match = _EMAIL_RE.search(text)
    if email_match:
        found["email"] = email_match.group(0)

    # Join digit groups split by separators ("0721 716 571") without
    # gluing the number to the words around it — compacting the whole
    # string turns "my number is 0721716571" into "...is0721716571",
    # where the pattern's leading boundary no longer matches.
    compact = re.sub(r"(?<=\d)[\s\-().]+(?=\d)", "", text)
    phone_match = _PHONE_RE.search(compact)
    if phone_match:
        found["phone"] = _normalise_phone(phone_match.group(0))

    # An explicit self-introduction is unambiguous, so it is read whether
    # or not we asked — and alongside a phone or email, since "I'm Lydia,
    # call me on 0721716571" carries both.
    intro = _NAME_INTRO_RE.search(text)
    if intro:
        candidate = _clean_name(intro.group(1))
        if candidate:
            found["full_name"] = candidate
            return found

    # A bare name has no signal of its own — "Lydia" and "maize" are
    # both single words — so from here on a name is only read when the
    # assistant asked for one in the preceding messages.
    if not _bot_asked_for_contact(conversation_history):
        return found

    # We asked for a name AND a number and got both at once: "John Doe ,
    # 0700123987 is the number". The phone was taken above; the name is
    # whatever leads the message before the digits start.
    #
    # Without this the name was silently dropped — the visitor typed it,
    # the bot read back only the number, and the lead reached the CRM as
    # "Chatwoot Visitor".
    if found:
        if "full_name" not in found:
            # Cut at whichever detail comes first. Splitting on "@"
            # would keep the email's local part ("lydia@..." leaving
            # "lydia"), so the cut is made at the start of the whole
            # match, not inside it.
            cut = len(text)
            if email_match:
                cut = min(cut, email_match.start())
            digit = re.search(r"\d", text)
            if digit:
                cut = min(cut, digit.start())

            candidate = _clean_name(text[:cut])
            if candidate:
                found["full_name"] = candidate

        return found

    # A question is not a name, however short.
    if "?" in text:
        return found

    candidate = _clean_name(text)
    if candidate:
        found["full_name"] = candidate

    return found


def _clean_name(raw: str) -> str | None:
    """Validate and title-case a name candidate, or None if it isn't one."""
    # Punctuation people leave around a name when typing it beside a
    # number: "John Doe ,", "Omari Tst-".
    words = [w.strip(".,!-–—:;") for w in raw.strip().split()]
    words = [w for w in words if w]

    # Drop the scaffolding around a number ("the number is 0700…") so
    # what remains is the name, or nothing.
    words = [w for w in words if w.lower() not in _NAME_FILLER]

    if not 1 <= len(words) <= MAX_NAME_WORDS:
        return None

    if " ".join(words).lower() in _NOT_NAMES:
        return None

    # A single filler word ("yes", "asante") is never a name even when it
    # arrives inside a longer candidate.
    if any(w.lower() in _NOT_NAMES for w in words):
        return None

    if not all(re.fullmatch(r"[A-Za-z'’\-]+", w) for w in words):
        return None

    return " ".join(w.capitalize() for w in words)


def acknowledgement(
    details: dict,
    language: str = "en",
    has_phone: bool = False,
) -> str:
    """What to say back once details are captured.

    Reads back what was heard, so a mistyped number is visible to the
    visitor while they can still correct it.

    The phone number is what makes a lead workable — the sales team calls
    people back, and an email reaches nobody in a field. So this keeps
    asking until a number is captured, and only then closes the
    conversation. Without that, a visitor who answers "how can we reach
    you?" with an email address gets thanked, told we will call, and the
    chat ends having never asked for a number. That is conversation 50
    exactly: she gave her email first.

    has_phone covers a number we already hold from Chatwoot (a pre-chat
    form, or an earlier conversation) rather than from this message.
    """
    name = details.get("full_name")
    captured = [details[k] for k in ("phone", "email") if k in details]
    phone_known = has_phone or "phone" in details

    if language == "sw":
        greeting = f"Asante {name}!" if name else "Asante!"
        body = (
            f"Nimepokea {' na '.join(captured)}"
            if captured
            else "Nimehifadhi maelezo yako"
        )

        if not phone_known:
            return f"{greeting} {body}. Ni nambari gani ya simu wakupigie?"

        return (
            f"{greeting} {body} — timu yetu itakupigia leo. Nitafunga "
            "mazungumzo haya sasa, lakini ombi lako liko kwao."
        )

    greeting = f"Thanks {name}!" if name else "Thanks!"
    body = (
        f"I've got {' and '.join(captured)}"
        if captured
        else "I've noted your details"
    )

    # Still no number: ask, and leave the conversation open for it. The
    # closing line would contradict the question, same as in handoff.
    if not phone_known:
        return f"{greeting} {body}. What number should the team call you on?"

    return (
        f"{greeting} {body} — our team will call you back today. I'll close "
        "this chat now, but your request is with them."
    )
