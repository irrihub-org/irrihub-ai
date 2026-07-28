import os
import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()

_pool: asyncpg.Pool | None = None


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError("DATABASE_URL is not set.")

    return url


async def _init_connection(conn: asyncpg.Connection) -> None:
    """
    Called by asyncpg for every new connection in the pool.
    Registers the pgvector codec so Python lists can be sent
    directly as `vector` query parameters.
    """
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    global _pool

    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=get_database_url(),
            min_size=1,
            max_size=10,
            init=_init_connection,
        )

    return _pool


async def fetch_top_k(
    embedding: list[float],
    k: int = 5,
):
    """
    Returns top-k chunks, each including:
      - distance: pgvector cosine distance (0=identical, 2=opposite),
        used by the confidence gate (see escalation.py).
      - source_filename: the human-readable document filename this
        chunk came from, used to build readable entries in the
        response's sources field (see main.py's _build_source_labels).
    """
    pool = await get_pool()

    query = """
    SELECT
        kd.id,
        kd.chunk_text,
        ks.filename AS source_filename,
        kd.embedding <=> $1 AS distance
    FROM kb_documents kd
    JOIN kb_sources ks
        ON kd.source_id = ks.id
    WHERE ks.is_active = true
    ORDER BY kd.embedding <=> $1
    LIMIT $2;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            query,
            embedding,
            k,
        )

    return [dict(row) for row in rows]


async def is_conversation_human_managed(conversation_id: str) -> bool:
    """
    Once a conversation has been escalated, a human owns it — the bot
    must not keep auto-replying underneath them. Checked at the start
    of every /chat call.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT is_human_managed FROM sessions WHERE conversation_id = $1",
            conversation_id,
        )
    return bool(row and row["is_human_managed"])


async def mark_conversation_human_managed(
    conversation_id: str, language: str
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (conversation_id, is_human_managed, language)
            VALUES ($1, true, $2)
            ON CONFLICT (conversation_id)
            DO UPDATE SET is_human_managed = true, last_active = now()
            """,
            conversation_id,
            language,
        )