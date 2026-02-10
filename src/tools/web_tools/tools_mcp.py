from fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INVALID_PARAMS, INTERNAL_ERROR
from tools.web_tools.web_search import web_search
from tools.web_tools.web_fetch import web_fetch

from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="web-mcp-server")

@mcp.tool(
    name="web_search",
    description=(
    """"
Web search using Google Serper (LangChain GoogleSerperAPIWrapper).

Parameters:
  - query: string
  - k: int (default 5)
  - domains: optional list of domain suffix filters, e.g. ["arxiv.org", "anthropic.com"]

Returns:
  { "query": ..., "results": [ {url,title,snippet,rank}, ... ] }
    """
    )
)
def tool_web_search(query: str, k: int = 5, domains: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        if not query or not query.strip():
            raise McpError(ErrorData(code=INVALID_PARAMS, message="query must be a non-empty string"))
        if not isinstance(k, int) or k <= 0:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="k must be a positive integer"))
        if domains is not None and not isinstance(domains, list):
            raise McpError(ErrorData(code=INVALID_PARAMS, message="domains must be a list of strings or null"))
        return web_search(query=query.strip(), k=int(k), domains=domains)
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"web_search failed: {e}"))


@mcp.tool(
    name="web_fetch",
    description=(
    """
Crawl a URL and return cleaned markdown (fit_markdown) via crawl4ai.

Parameters:
  - url: string (target URL)

Returns:
  markdown string
    """
    )
)
async def tool_web_fetch(url: str) -> str:
    try:
        return await web_fetch(url=url)
    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"web_fetch failed: {e}"))


if __name__ == '__main__':
    host = "0.0.0.0"
    port = 10000
    mcp.run(transport="http", host=host, port=port)