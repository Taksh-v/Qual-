# Qual MCP Integration

## Run modes

- STDIO server:
  - `python run_mcp_stdio.py`
- Streamable HTTP server:
  - `python run_mcp_http.py`
  - Health: `GET /health`
  - MCP endpoint: `/mcp`

## Authentication

- Read/admin keys are configured with:
  - `MCP_READ_KEYS`
  - `MCP_ADMIN_KEYS`
- If both are unset, MCP runs in dev mode.
- If unset but `API_KEYS` is set, MCP falls back to `API_KEYS`.

## Admin execution model

Admin actions are protected by two-step execution and idempotency:

1. Call `prepare_admin_action_tool` with action + params.
2. Call `execute_admin_action_tool` with:
   - `action`
   - `confirmation_token`
   - `idempotency_key`

Available admin actions:

- `reload_index`
- `scheduler_job`
- `quality_gate`
- `rag_eval`
- `vector_audit`
- `data_quality_audit`

## External MCP client mode

Configure outbound MCP servers with `MCP_CLIENT_SERVERS_JSON`.

Example:

```json
{
  "internal-tools": {
    "transport": "streamable-http",
    "url": "http://localhost:9000/mcp"
  },
  "local-stdio": {
    "transport": "stdio",
    "command": "python",
    "args": ["path/to/server.py"]
  }
}
```

Use tools:

- `list_external_mcp_servers_tool`
- `call_external_mcp_tool`

## Runtime enrichment hooks

You can optionally enrich core runtime responses by calling a configured external MCP server/tool:

- RAG runtime (`rag/query.py`):
  - `RAG_ENABLE_MCP_ENRICHMENT=1`
  - `RAG_MCP_SERVER=<server-name>`
  - `RAG_MCP_TOOL=<tool-name>`
- Macro runtime (`intelligence/macro_engine.py`):
  - `MACRO_ENABLE_MCP_ENRICHMENT=1`
  - `MACRO_MCP_SERVER=<server-name>`
  - `MACRO_MCP_TOOL=<tool-name>`

Both hooks are fail-safe: if the external call fails, core response generation continues normally.
