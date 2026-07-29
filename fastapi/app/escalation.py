"""
Confidence gate + escalation logging 

Computes a confidence score from pgvector retrieval distance and
decides the route:

  Band     Score        Action                          User sees
  High     >= 0.65      Respond with KB-grounded answer  Direct answer with source citation
  Medium   0.20 - 0.64  Respond with KB-grounded answer  Direct answer with source citation
                                                          (same as High - no caveat)
  Low      < 0.20       Immediate escalation, LLM        "Let me connect you with our
                         skipped                          support team."



"""

from typing import Literal

from app.db import get_pool

Route = Literal["respond", "escalate", "deterministic"]

NO_MATCH_DISTANCE_FLOOR = 1.6  
HIGH_CONFIDENCE_THRESHOLD = 0.65
MEDIUM_CONFIDENCE_THRESHOLD = 0.20


def distance_to_confidence(distance: float) -> float:
    confidence = 1 - (distance / 2)
    return max(0.0, min(1.0, confidence))


def determine_route(top_distance: float | None) -> tuple[float, Route, bool]:
    
    if top_distance is None:
        return 0.0, "escalate", False

    if top_distance > NO_MATCH_DISTANCE_FLOOR:
        confidence = distance_to_confidence(top_distance)
        return confidence, "escalate", False

    confidence = distance_to_confidence(top_distance)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return confidence, "respond", False

    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return confidence, "respond", False

    return confidence, "escalate", False


async def log_kb_gap(
    query_text: str,
    confidence: float | None = None,
    channel: str | None = None,
    failure_reason: str | None = None,
) -> None:

  
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO kb_gaps (query_text, channel, confidence, failure_reason)
            VALUES ($1, $2, $3, $4)
            """,
            query_text,
            channel,
            confidence,
            failure_reason,
        )


