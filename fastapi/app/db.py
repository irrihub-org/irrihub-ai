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
    pool = await get_pool()

    query = """
    SELECT
        kd.id,
        kd.chunk_text
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