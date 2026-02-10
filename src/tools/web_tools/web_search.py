from urllib.parse import urlparse

from typing import Optional, Dict, Any

def build_serper_wrapper(k: int):
    from langchain_community.utilities import GoogleSerperAPIWrapper

    return GoogleSerperAPIWrapper(k=k)

def web_search(query: str, k: int, domains: Optional[list[str]] = None) -> Dict[str, Any]:
    """
        Search the web via Serper and return a normalized results list.

        Args:
            query: Search query
            k: Max number of results
            domains: Optional list of allowed domains (e.g. ["anthropic.com", "arxiv.org"])
    """
    searcher = build_serper_wrapper(k)
    raw = searcher.results(query)
    organic = raw.get("organic", []) if isinstance(raw, dict) else []

    results: list[dict] = []
    seen: set[str] = set()

    for idx, item in enumerate(organic, start=1):
        item = item or {}
        url = item.get("link") or ""
        if not url or url in seen:
            continue
        seen.add(url)

        if domains:
            netloc = urlparse(url).netloc.lower()
            if not any(netloc.endswith(d.lower()) for d in domains):
                continue

        results.append(
            {
                "url": url,
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "rank": item.get("position") or idx,
            }
        )
        if len(results) >= k:
            break

    return {
        "query": query,
        "results": results,
    }