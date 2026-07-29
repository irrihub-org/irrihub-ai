
import re

VALID_INTENTS = {
    "greeting",
    "hardware_fault",
    "escalation_request",
    "product_inquiry",
    "general_query",
}

GREETING_PATTERNS = [
    r"\bhabari\b", r"\bhujambo\b", r"\bmambo\b", r"\bsasa\b",
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b",
    r"\bgood afternoon\b", r"\bgood evening\b",
]

ESCALATION_PATTERNS = [
    r"\btalk to (a |an )?(human|agent|person|someone)\b",
    r"\bspeak to (a |an )?(human|agent|person|someone)\b",
    r"\bcustomer (service|support)\b",
    r"\breal person\b",
    r"\bongea na mtu\b",
    r"\bnataka msaada wa mtu\b",
    r"\bnisaidie mtu\b",
]

# --- Hardware fault: patterns grouped by category ---
HARDWARE_FAULT_CATEGORIES: dict[str, list[str]] = {
    "pump_troubleshooting": [
        r"\bpump\b.*\b(not working|broken|won'?t start|doesn'?t start|no power)\b",
        r"\b(not working|broken|won'?t start|doesn'?t start)\b.*\bpump\b",
        r"\bhaifanyi kazi\b", r"\bhaiwaki\b", r"\bhaianzi\b", r"\bhaina nguvu\b",
        r"\bno power\b", r"\bime\s?hang\b",
    ],
    "drip_clogging": [
        r"\bclog(ging|ged)?\b", r"\bemitter(s)?\b.*\b(block|clog)",
        r"\bno flow\b", r"\blow flow\b", r"\bhaitoi maji\b",
    ],
    "leak": [
        r"\bleak(ing|s|age)?\b", r"\binavuja\b",
    ],
    "controller_fault": [
        r"\bcontroller\b", r"\berror\b", r"\bfault\b",
    ],
}

PRODUCT_INQUIRY_CATEGORIES: dict[str, list[str]] = {
    "pricing": [
        r"\bprice\b", r"\bcost\b", r"\bhow much\b", r"\bbei\b", r"\bgharama\b",
    ],
    "availability": [
        r"\bavailable\b", r"\bin stock\b", r"\bdo you (sell|have)\b",
    ],
    "purchasing": [
        r"\bbuy\b", r"\bpurchase\b", r"\border\b", r"\bkununua\b",
        r"\bnataka kununua\b", r"\bcatalog(ue)?\b",
    ],
}


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _first_matching_category(categories: dict[str, list[str]], text: str) -> str:
    """Returns the first category whose patterns match, or '' if none do."""
    for category, patterns in categories.items():
        if _matches_any(patterns, text):
            return category
    return ""


def classify_intent(message: str) -> tuple[str, str]:
    """
    Classify a user message into (intent, category).

    intent is always one of VALID_INTENTS. category is a sub-type
    within that intent, or '' if no specific sub-type was detected
    (or the intent has no meaningful sub-types, e.g. greeting).

    Checks in priority order: escalation requests and greetings are
    usually unambiguous and short, so check those first; hardware
    faults next since they're time-sensitive; product inquiries
    after; otherwise fall back to general_query.
    """
    text = message.lower().strip()

    if not text:
        return "general_query", ""

    if _matches_any(ESCALATION_PATTERNS, text):
        return "escalation_request", ""

    # Only classify as a greeting if the message is short and doesn't
    # also contain a real question - "hello, my pump is broken" should
    # be hardware_fault, not greeting.
    if len(text.split()) <= 4 and _matches_any(GREETING_PATTERNS, text):
        return "greeting", ""

    hardware_category = _first_matching_category(HARDWARE_FAULT_CATEGORIES, text)
    if hardware_category:
        return "hardware_fault", hardware_category

    product_category = _first_matching_category(PRODUCT_INQUIRY_CATEGORIES, text)
    if product_category:
        return "product_inquiry", product_category

    return "general_query", ""
