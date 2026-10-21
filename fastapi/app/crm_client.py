"""
Send escalated conversations to the CRM as leads.

The workflow this closes: a farmer reaches out on the chatbot, the bot
hands off to a human, and the lead lands in the CRM so the sales team can
follow up — including when nobody was online in Chatwoot at the time.

Why this lives here rather than in n8n. The n8n workflow was the obvious
home, and the original design put an HTTP node on the escalate branch.
Two things argued against it: the live workflow has drifted from the
version in source control, so anything built there is fragile against
edits nobody can see; and the CRM call needs the same data the assistant
has just assembled (crm_payload), which n8n would have to thread back out
of the response. Calling from here keeps the lead logic beside the logic
that produces it, in a repo with tests.

Delivery is fire-and-forget. The farmer is waiting on the reply this
request is building, and a slow or down CRM must never delay it or break
the escalation. A lost lead record is recoverable — the conversation is
still labelled and assigned in Chatwoot. A farmer left hanging is not.
"""

import asyncio
import os

import httpx

CRM_BASE_URL = os.getenv("CRM_BASE_URL", "http://irrihub-app")
CRM_LEAD_TOKEN = os.getenv("CHATWOOT_LEAD_WEBHOOK_TOKEN")

_ENDPOINT = "/api/leads/chatwoot"

# asyncio keeps only a weak reference to a running task, so a
# fire-and-forget one can be garbage-collected mid-flight and the lead
# silently lost. Holding a strong reference until it finishes is the
# documented way to prevent that.
_IN_FLIGHT: set = set()
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0

# Chatwoot's channel strings mapped to what the CRM records. Anything
# unrecognised is sent through as-is: the CRM validates, and a new
# channel appearing in logs is better than one silently relabelled.
_CHANNEL_MAP = {
    "Channel::WebWidget": "web",
    "Channel::Whatsapp": "whatsapp",
    "Channel::Api": "api",
    "web": "web",
    "whatsapp": "whatsapp",
}


def is_configured() -> bool:
    """Whether a lead can be sent at all."""
    return bool(CRM_LEAD_TOKEN)


def normalise_channel(channel: str | None) -> str:
    return _CHANNEL_MAP.get((channel or "").strip(), (channel or "web").strip())


def build_lead(
    conversation_id: str,
    crm_payload: dict,
    channel: str | None = None,
    reason: str | None = None,
    language: str = "en",
) -> dict:
    """The payload the CRM's /api/leads/chatwoot expects.

    contact_id is the Chatwoot conversation id rather than the contact
    id, because the assistant is not told the latter. That is a weaker
    dedup key — the same person returning in a new conversation would
    look like a new lead — but only for visitors who never give a phone
    number. The CRM dedups on phone first, so anyone who does is matched
    across conversations and channels regardless of this id.
    """
    return {
        "contact_id": str(conversation_id),
        "conversation_id": str(conversation_id),
        "full_name": crm_payload.get("full_name"),
        "phone": crm_payload.get("phone"),
        "email": crm_payload.get("email"),
        "actor_type": "farmer",
        "language": language,
        "channel": normalise_channel(channel),
        "reason": reason or "Escalated to a human agent by the AI assistant",
        "summary": crm_payload.get("summary"),
    }


async def _post(lead: dict) -> None:
    headers = {
        "Authorization": f"Bearer {CRM_LEAD_TOKEN}",
        "Content-Type": "application/json",
    }
    url = f"{CRM_BASE_URL.rstrip('/')}{_ENDPOINT}"

    last_error: str | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                response = await client.post(url, json=lead, headers=headers)

            if response.status_code < 300:
                body = response.json()
                print(
                    f"[crm-lead] created party={body.get('party_id')} "
                    f"conversation={lead['conversation_id']} "
                    f"phone={'yes' if lead.get('phone') else 'NO'}"
                )
                return

            # 4xx is our mistake — a bad token or a payload the CRM
            # rejects. Retrying sends the identical request, so stop and
            # make the reason visible instead.
            if response.status_code < 500:
                print(
                    f"[crm-lead] REJECTED {response.status_code} "
                    f"conversation={lead['conversation_id']}: {response.text[:300]}"
                )
                return

            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 - never break the escalation
            last_error = repr(exc)

        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    print(
        f"[crm-lead] FAILED after {_MAX_ATTEMPTS} attempts "
        f"conversation={lead['conversation_id']}: {last_error}"
    )


def send_lead(
    conversation_id: str,
    crm_payload: dict,
    channel: str | None = None,
    reason: str | None = None,
    language: str = "en",
) -> None:
    """Queue a lead for delivery and return immediately.

    Returns None, not an awaitable: callers are on the path of the
    farmer's reply and must not wait. Failures are logged, never raised.
    """
    if not is_configured():
        print(
            "[crm-lead] CHATWOOT_LEAD_WEBHOOK_TOKEN is not set — lead for "
            f"conversation={conversation_id} was NOT sent"
        )
        return

    lead = build_lead(conversation_id, crm_payload, channel, reason, language)

    # Check for a loop BEFORE building the coroutine. Calling _post()
    # first and letting create_task fail leaves an un-awaited coroutine
    # behind, which Python warns about and which hides the real problem.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (a synchronous caller, or a test). Nothing to
        # schedule onto, and blocking here would defeat the point.
        print(
            f"[crm-lead] no event loop; lead for conversation={conversation_id} "
            "was not sent"
        )
        return

    task = asyncio.create_task(_post(lead))
    _IN_FLIGHT.add(task)
    task.add_done_callback(_IN_FLIGHT.discard)
