"""
Tests for the lead summary.

The summary is a convenience attached to an escalation that matters more
than it does, so the behaviour that counts most is what happens when the
LLM call fails: it must return None quietly, never raise.
"""

import pytest

from app import lead_summary
from app.lead_summary import MAX_SUMMARY_CHARS, summarise_request


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the LLM call with a scripted reply."""

    def _install(reply=None, raises=None):
        async def _fake(prompt, system_prompt=None, models=None):
            if raises is not None:
                raise raises
            return reply

        monkeypatch.setattr(lead_summary, "generate_response", _fake)

    return _install


@pytest.mark.asyncio
async def test_returns_the_summary(fake_llm):
    fake_llm("Wants a quote for a drip kit, asked to be called at 9am.")
    assert await summarise_request("[USER] : I need a quote") == (
        "Wants a quote for a drip kit, asked to be called at 9am."
    )


@pytest.mark.asyncio
async def test_an_llm_failure_never_breaks_the_escalation(fake_llm):
    """The escalation must proceed even with no summary."""
    fake_llm(raises=RuntimeError("OPENROUTER_API_KEY is not set."))
    assert await summarise_request("[USER] : I need a quote") is None


@pytest.mark.asyncio
async def test_no_history_means_no_call(fake_llm):
    fake_llm(raises=AssertionError("should not have been called"))
    assert await summarise_request(None) is None
    assert await summarise_request("   ") is None


@pytest.mark.asyncio
async def test_the_no_request_sentinel_becomes_none(fake_llm):
    """An empty summary is worse than none — it looks like real content."""
    fake_llm("No specific request stated.")
    assert await summarise_request("[USER] : hi") is None


@pytest.mark.asyncio
async def test_whitespace_and_quotes_are_tidied(fake_llm):
    fake_llm('  "Wants   a quote\nfor drip."  ')
    assert await summarise_request("x") == "Wants a quote for drip."


@pytest.mark.asyncio
async def test_a_long_summary_is_truncated_to_fit_the_crm_column(fake_llm):
    fake_llm("word " * 400)
    got = await summarise_request("x")
    assert len(got) <= MAX_SUMMARY_CHARS


@pytest.mark.asyncio
async def test_an_empty_reply_is_none(fake_llm):
    fake_llm("")
    assert await summarise_request("[USER] : I need a quote") is None
