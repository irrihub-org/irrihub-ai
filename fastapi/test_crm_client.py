"""
Tests for the CRM lead client.

The behaviour that matters most is what happens when the CRM is down or
misconfigured: the farmer is waiting on the reply this request is
building, so a failure must be logged and swallowed, never raised and
never blocking.
"""

import pytest

from app import crm_client
from app.crm_client import build_lead, is_configured, normalise_channel, send_lead


def test_the_payload_matches_what_the_crm_validates():
    lead = build_lead(
        "54",
        {
            "full_name": "Omari Tst",
            "phone": "+254701112345",
            "email": "omari@example.com",
            "summary": "Wants a quote for a drip kit.",
        },
        channel="web",
        language="en",
    )

    assert lead == {
        "contact_id": "54",
        "conversation_id": "54",
        "full_name": "Omari Tst",
        "phone": "+254701112345",
        "email": "omari@example.com",
        "actor_type": "farmer",
        "language": "en",
        "channel": "web",
        "reason": "Escalated to a human agent by the AI assistant",
        "summary": "Wants a quote for a drip kit.",
        "transcript": None,
    }


def test_the_transcript_travels_with_the_lead():
    """The chat auto-resolves ten minutes after the handoff, so this is
    often the only record of what was said."""
    chat = "[USER] : hi\n[ASSISTANT] : Hello!\n[USER] : I need a quote"
    lead = build_lead("54", {"transcript": chat})

    assert lead["transcript"] == chat


def test_an_anonymous_escalation_still_produces_a_valid_lead():
    """Most escalations have nothing but a conversation id. The CRM
    accepts that — it dedups on (source, external_id) when there is no
    phone — and a lead nobody can call is still a lead somebody can
    reply to in Chatwoot."""
    lead = build_lead("54", {})

    assert lead["contact_id"] == "54"
    assert lead["full_name"] is None
    assert lead["phone"] is None
    assert lead["summary"] is None


def test_chatwoot_channel_strings_are_mapped():
    assert normalise_channel("Channel::WebWidget") == "web"
    assert normalise_channel("Channel::Whatsapp") == "whatsapp"
    assert normalise_channel("whatsapp") == "whatsapp"
    assert normalise_channel(None) == "web"


def test_an_unknown_channel_passes_through_rather_than_being_relabelled():
    """The CRM validates the value. A new inbox showing up in its logs
    is better than one silently recorded as a web chat."""
    assert normalise_channel("Channel::Telegram") == "Channel::Telegram"


def test_conversation_id_is_used_as_the_dedup_key():
    """The assistant is not told the Chatwoot contact id. That is a
    weaker key — only for visitors who never give a phone number, since
    the CRM matches on phone first."""
    lead = build_lead("99", {"phone": "+254700000000"})

    assert lead["contact_id"] == "99"
    assert lead["conversation_id"] == "99"


def test_nothing_is_sent_when_the_token_is_missing(monkeypatch, capsys):
    """Must warn loudly rather than fail silently — a missing token
    means every lead is being dropped."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", None)

    send_lead("54", {"phone": "+254700000000"})

    assert "not set" in capsys.readouterr().out


def test_is_configured_reflects_the_token(monkeypatch):
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", None)
    assert is_configured() is False

    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")
    assert is_configured() is True


def test_send_lead_returns_immediately_without_an_event_loop(monkeypatch, capsys):
    """send_lead is called on the path of the farmer's reply. With no
    loop to schedule onto it must return, not block or raise."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")

    assert send_lead("54", {"phone": "+254700000000"}) is None
    assert "no event loop" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_task_is_held_until_it_finishes(monkeypatch):
    """asyncio keeps only a weak reference to a running task, so a
    fire-and-forget one can be garbage-collected mid-flight and the lead
    silently lost. It must be held until done, then released."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")

    posted = []

    async def _fake_post(lead):
        posted.append(lead)

    monkeypatch.setattr(crm_client, "_post", _fake_post)

    send_lead("54", {"phone": "+254700000000"})

    assert len(crm_client._IN_FLIGHT) == 1, "task must be held while running"

    # Two yields: one for the task body, one for the done-callback that
    # releases it — add_done_callback is scheduled, not immediate.
    import asyncio as _asyncio

    await _asyncio.sleep(0)
    await _asyncio.sleep(0)

    assert posted, "the lead should have been posted"
    assert not crm_client._IN_FLIGHT, "the task must be released once done"


@pytest.mark.asyncio
async def test_a_crm_outage_is_swallowed(monkeypatch, capsys):
    """The escalation must survive the CRM being unreachable."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")
    monkeypatch.setattr(crm_client, "_MAX_ATTEMPTS", 1)

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(crm_client.httpx, "AsyncClient", Boom)

    await crm_client._post(build_lead("54", {}))

    assert "FAILED" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_rejected_payload_is_not_retried(monkeypatch, capsys):
    """A 4xx is our bug — a bad token or a payload the CRM refuses.
    Retrying sends the identical request, so stop and surface why."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")
    attempts = {"n": 0}

    class Rejecting:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            attempts["n"] += 1

            class R:
                status_code = 422
                text = '{"error":"invalid payload"}'

            return R()

    monkeypatch.setattr(crm_client.httpx, "AsyncClient", Rejecting)

    await crm_client._post(build_lead("54", {}))

    assert attempts["n"] == 1
    assert "REJECTED 422" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_success_is_logged_with_whether_a_phone_was_captured(
    monkeypatch, capsys
):
    """Phone presence is the thing the sales team cares about, so it
    belongs in the log line rather than needing a CRM lookup."""
    monkeypatch.setattr(crm_client, "CRM_LEAD_TOKEN", "a-token")

    class Ok:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            class R:
                status_code = 200

                @staticmethod
                def json():
                    return {"party_id": 143, "status": "ok"}

            return R()

    monkeypatch.setattr(crm_client.httpx, "AsyncClient", Ok)

    await crm_client._post(build_lead("54", {"phone": "+254700000000"}))
    out = capsys.readouterr().out

    assert "party=143" in out
    assert "phone=yes" in out

    await crm_client._post(build_lead("55", {}))
    assert "phone=NO" in capsys.readouterr().out
