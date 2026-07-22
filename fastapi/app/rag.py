from typing import List, Dict, Any
from app.db import fetch_top_k


def embed_query(query: str, model) -> List[float]:
    """
    Convert user query into embedding vector.
    """
    return model.encode(query).tolist()


async def retrieve_top_k_by_embedding(embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant KB chunks from pgvector DB, given a
    precomputed embedding vector.
    """
    results = await fetch_top_k(embedding, k=k)
    return results


async def retrieve_top_k(query: str, model, k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience wrapper used by the API layer: takes the raw query string
    and the embedding model, embeds it, then retrieves the top-k chunks.
    """
    embedding = embed_query(query, model)
    return await retrieve_top_k_by_embedding(embedding, k=k)


async def rag_pipeline(query: str, model, k: int = 5) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
    query → embedding → vector search → top-k results

    """
    results = await retrieve_top_k(query, model, k=k)

    return {
        "query": query,
        "results": results
    }