# DeepTB Pilot - AI Coding Agent Instructions

## Project Overview

DeepTB Pilot is an AI agent for the DeePTB materials science software, combining a React/TypeScript frontend with a Python backend that uses LLMs, Model Context Protocol (MCP), and Bohr Agent SDK (Google ADK).

**Critical Architectural Understanding:**
- **Two-process architecture**: MCP Tools Server (`dptb-tools`) and Main App (`dptb-pilot`) must run concurrently
- **Same environment requirement**: DeePTB package and dptb-pilot MUST share the same Python environment
- **MCP tool interception**: Tools in `target_tools` pause execution for user parameter confirmation via `tool_modify_guardrail`

## Running the Application

```bash
# Terminal 1: Start MCP tools server
dptb-tools  # Runs on port 50002

# Terminal 2: Start main application (auto-launches frontend)
dptb-pilot  # Backend on 8000, frontend on 50001
```

Configuration via `.env` (see `env.example`):
- `LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY` - LLM provider configuration (uses LiteLLM)
- `MP_API_KEY` - Materials Project API key (optional)
- `MCP_TOOLS_URL` - Must match where `dptb-tools` is running
- `WORK_ROOT` - Session workspace directory (default: `./workspace`)

## Architecture & Data Flow

### Backend Services (Python)

**1. Main Application** (`backend/better_aim/`)
- `react_main.py`: CLI entry point, orchestrates frontend + backend startup
- `react_host.py`: FastAPI server with WebSocket support for streaming responses
- `agent.py`: Creates `LlmAgent` using Google ADK + LiteLLM, connects to MCP tools
- `adjustable_session_service.py`: Extends ADK's session service to support event manipulation (undo)
- `tool_modify_guardrail.py`: Callback that intercepts MCP tool calls, sends schemas to frontend for user approval

**2. MCP Tools** (`backend/dptb_agent_tools/`)
- `main.py`: MCP server entry point (uses `fastmcp` by default)
- `init_mcp.py`: Initializes MCP server instance based on `DPTB_AGENT_MODEL` env var
- `modules/`: MCP tool implementations decorated with `@mcp.tool()`
  - `workspace_tool.py`: File management in session workspace
  - `mp_tool.py`: Materials Project database search
  - `rag_tool.py`: ChromaDB-based semantic search over DeePTB docs
  - `config_tool.py`, `sk_baseline_model.py`, `sk_predict.py`: DeePTB integration
  - `visualize_tool.py`, `visualize_bz_tool.py`, `band.py`: Structure/band visualization

**Tool Guardrail Pattern**: Tools in `target_tools` trigger async pause → frontend displays editable schema → user confirms → backend resumes with `modified_args_store` values

### Frontend (React + TypeScript)

- `src/App.tsx`: Router setup (Login → Chat)
- `src/contexts/AppContext.tsx`: Global state (session, files, WebSocket)
- `src/services/api.ts`: Axios-based API client (uses `/api` prefix for Vite proxy)
- `src/components/Chat/`: Main interface with message streaming, 3Dmol.js structure viewer
- `src/components/SessionPanel/`, `FilePanel/`, `ParamPanel/`: Sidebar UI

**WebSocket Message Flow**: Backend sends JSON messages with `type` field (`text`, `tool_call`, `structure`, `band`, `error`) → Chat component renders accordingly

### Session & Workspace Management

Each session has:
- Unique ID (32-char random string)
- Workspace directory: `{WORK_ROOT}/{session_id}/`
- Persistent storage: `{workspace_dir}/sessions.json` (chat history, metadata)
- File uploads: `{workspace_dir}/files/`

Files are uploaded via multipart/form-data to `/upload` endpoint, stored in session workspace.

## Key Development Patterns

### Adding New MCP Tools

1. Create module in `backend/dptb_agent_tools/modules/`
2. Import `mcp` from `dptb_agent_tools.init_mcp`
3. Define tool with decorator:
   ```python
   @mcp.tool()
   def my_tool(param: str) -> str:
       """Tool description for LLM."""
       # Implementation
   ```
4. Explicitly import in `dptb_agent_tools/main.py` (line 18-25)
5. Restart `dptb-tools` to register

### Frontend API Integration

API calls use relative `/api` prefix (proxied by Vite to backend). Common endpoints:
- `POST /login`: Initialize session
- `POST /chat`: Send message (returns text or streaming via WS)
- `GET /ws/{session_id}`: WebSocket for streaming responses
- `POST /upload`: File upload to session workspace
- `GET /workspace/{session_id}`: List workspace files
- `POST /modify_params`: Submit modified tool parameters

### RAG Knowledge Base

DeePTB documentation is embedded in ChromaDB:
- Build: `python backend/dptb_agent_tools/build_knowledge_base.py`
- Storage: `backend/dptb_agent_tools/data/chroma_db/`
- Search: `search_knowledge_base(query)` tool (uses `sentence-transformers/all-MiniLM-L6-v2`)

**Source Priority**: Code files, then markdown docs, then PDFs in `data/deeptb_knowledge/`

## Testing & Debugging

**No formal test suite exists.** Manual testing workflow:
1. Check MCP tools: `dptb-tools --help` (lists available transports)
2. Backend health: `curl http://localhost:8000/health`
3. WebSocket test: Browser DevTools → Network → WS → observe message flow
4. Agent logs: Check terminal output for `--- Callback:` messages (tool guardrail traces)

**Common Issues**:
- "I don't have that tool": MCP server not running or wrong `MCP_TOOLS_URL`
- Empty responses: Check LLM API credentials in `.env`
- Frontend proxy errors: Verify `BACKEND_HOST` matches where FastAPI is accessible

## Code Style Conventions

- **Python**: No strict formatter, but follow existing style (4-space indent, snake_case)
- **TypeScript**: React functional components, Ant Design for UI, avoid class components
- **Comments**: Chinese comments common in original code, English acceptable for new code
- **Imports**: Explicit imports preferred over dynamic loading (see `dptb_agent_tools/main.py`)

## Dependencies & Environment

**Python**: Requires 3.10+ (uses type hints, match statements)
**Key packages**: `mcp>=1.9.0`, `bohr-agent-sdk`, `LiteLlm`, `chromadb`, `pymatgen`, `mp-api`
**Frontend**: Vite + React 18 + Ant Design 5 + 3Dmol.js

Install in editable mode: `pip install -e .` (installs `dptb-pilot` and `dptb-tools` scripts)

## Integration Points

- **DeePTB**: Tools invoke DeePTB CLI or Python API (must be in same venv)
- **Materials Project**: `mp_tool.py` uses `MPRester` (requires `MP_API_KEY`)
- **LiteLLM**: Abstracts LLM providers (OpenAI, Azure, DeepSeek, etc.) via unified interface
- **Google ADK**: Agent framework managing conversation state, tool orchestration

## Critical Files to Reference

- `backend/better_aim/react_host.py`: Backend API server and WebSocket handler (622 lines)
- `backend/better_aim/tool_modify_guardrail.py`: Tool interception logic
- `backend/dptb_agent_tools/init_mcp.py`: MCP server initialization (model selection)
- `frontend/src/services/api.ts`: Frontend API client
- `pyproject.toml`: Package metadata, dependencies, console scripts
