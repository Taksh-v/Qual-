# GEMINI.md - Project Mandates & Persistent Context

## Project Overview
**Finance AI System (Qual)**: A news-driven financial intelligence RAG pipeline designed for multi-step macro reasoning, sentiment analysis, and market outlook generation.

## Core Architecture
- **Data Pipeline**: `Ingestion` → `Cleaning` → `Chunking` → `Embedding` → `Vector Index (FAISS)`.
- **Query Layer**: RAG retrieval (`rag/query.py`) + LLM generation.
- **Intelligence Layer**: Macro reasoning, trend analysis, and sector mapping.
- **API Layer**: FastAPI-based endpoints for `/ask` and `/intelligence`.

## Foundational Mandates
- **Surgical Updates**: Prioritize targeted changes. Do not perform unrelated refactoring.
- **Verification First**: Always reproduce bugs with a test case before fixing. All features require automated tests.
- **Type Safety**: Use Python type hints for all new functions and classes.
- **Documentation**: Maintain comprehensive docstrings (Google or NumPy style) and update relevant `.md` files in `docs/`.
- **Tooling**: Prefer local ecosystem tools (e.g., `pytest`, `ruff`, `mypy`) for validation.

## Persistent Strategy: System Renovation Plan
The project is currently undergoing a major restructuring of the response format system (see `SYSTEM_RENOVATION_PLAN.md`). All new "Intelligence" features must align with this plan:
1. **Centralized Schema**: Use `intelligence/response_schema.py` for all response types.
2. **Builder Pattern**: Construct responses via `intelligence/response_builder.py`.
3. **Template-Driven**: Move all prompt text to `intelligence/prompt_templates.py`.
4. **Validation**: Enforce quality through `intelligence/response_validator.py`.

## Technical Stack
- **Language**: Python 3.10+
- **Vector DB**: FAISS
- **Embeddings**: Local Ollama (`nomic-embed-text`)
- **LLM**: Local Ollama or configured API
- **Web Framework**: FastAPI
- **Search**: BM25 + Vector Hybrid (planned/implemented)

## Coding Conventions
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Citations**: Standardized source tracking using `[Sx]` format.
- **Emojis**: Use standardized indicators (e.g., `▲`, `▼`, `●`) as defined in `intelligence/writing_style.py`.
