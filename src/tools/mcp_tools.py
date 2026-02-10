from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def http_connection(
        *,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        auth: Any = None
) -> dict[str, Any]:
    """
    Build an HTTP (streamable-http) connection config.
    """
    conn: dict[str, Any] = {"transport": "http", "url": url}
    if headers:
        conn["headers"] = dict(headers)
    if auth is not None:
        conn["auth"] = auth
    return conn


def build_mcp_connections(
        *,
        server_name: str = "web_mcp_server",
        # ---- http only ----
        url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        auth: Any = None,
) -> dict[str, Any]:
    if not url:
        raise ValueError("url must be provided when transport='http'. Example: http://localhost:8000/mcp")
    return {
        server_name: http_connection(url=url, headers=headers, auth=auth)
    }


async def get_tools(
        connections: Mapping[str, Any],
        *,
        tool_name_prefix: bool = False,
        stateful: bool = False,
):
    server_name = list(connections.keys())[0]
    client = MultiServerMCPClient(dict(connections), tool_name_prefix=tool_name_prefix)
    if stateful:
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)
            return tools

    tools = await client.get_tools()
    return tools


async def get_web_tools(stateful: bool = False):
    conn = build_mcp_connections(
        server_name="web-mcp-server",
        url="http://localhost:10000/mcp",
    )
    return await get_tools(connections=conn, stateful=stateful)


async def get_docs_tools(stateful: bool = False):
    conn = build_mcp_connections(
        server_name="docs-mcp-server",
        url="http://localhost:10010/mcp",
    )
    return await get_tools(connections=conn, stateful=stateful)