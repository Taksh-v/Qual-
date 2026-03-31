from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Mock External MCP")


@mcp.tool()
def external_macro_context(question: str) -> dict[str, str]:
    return {"summary": f"External macro context: {question}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
