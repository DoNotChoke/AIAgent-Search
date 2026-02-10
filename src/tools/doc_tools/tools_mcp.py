import json
from fastmcp import FastMCP
from mcp.shared.exceptions import McpError

from sentence_transformers import SentenceTransformer

from storage.config import config
from storage.searcher import NeuralSearcher

from typing import List, Dict, Any

model = SentenceTransformer(config.model_path)
searcher = NeuralSearcher(config, model)

mcp = FastMCP(name="docs-mcp-server")

@mcp.tool(
    name="docs_search",
    description=("""
Search relevant document chunks for a query using Feast(Milvus) + MinIO.
Returns JSON string with top-k hits including chunk text.

Args:
    query: user query string
    k: number of results to return
    """
    )
)
def docs_search(query: str, k: int) -> str:
    if not query or not query.strip():
       raise McpError("Query cannot be empty")
    k = int(k) if k else 5
    if k <= 0:
        k = 5

    hits = searcher.search(query.strip(), k=k)

    results: List[Dict[str, Any]] = []
    for h in hits:
        text = (h.text or "")

        results.append({
            "score": h.score,
            "chunk_id": h.chunk_id,
            "s3_uri": h.s3_uri,
            "url": (h.fields or {}).get("url"),
            "doc_id": (h.fields or {}).get("doc_id"),
            "retrieved_at": (h.fields or {}).get("retrieved_at"),
            "text": text,
        })

    return json.dumps(
        {"query": query, "k": k, "results": results},
        ensure_ascii=False,
    )

if __name__ == '__main__':
    host = "0.0.0.0"
    port = 10010
    mcp.run(transport="http", host=host, port=port)