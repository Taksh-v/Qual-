# System Renovation & Response Format Restructuring Plan

## Executive Summary
This document outlines a comprehensive restructuring of the writing and response format system for the News Intelligence RAG API. The goal is to create a **centralized, reusable, and maintainable response framework** while preserving the existing architecture.

---

## PART 1: CURRENT STATE ANALYSIS

### 1.1 System Overview
**Type**: News-driven financial intelligence RAG pipeline  
**Current Architecture**:
- **Data Pipeline**: News ingestion → Cleaning → Chunking → Embedding → Vector Index (FAISS)
- **Query Layer**: RAG retrieval + LLM generation
- **Intelligence Layer**: Macro reasoning, sentiment analysis, trend analysis, regime detection
- **API Layer**: FastAPI with authentication, rate limiting, response caching
- **Key Components**: 
  - `rag/query.py` - Core RAG pipeline
  - `intelligence/macro_reasoner.py` - Prompt building & response construction
  - `intelligence/macro_engine.py` - Orchestration
  - `intelligence/response_enhancer.py` - Post-processing validation
  - `intelligence/market_context.py` - Market context building
  - `api/app.py` - API endpoints

### 1.2 Current Response Format Issues

#### Issue #1: Inconsistent Response Structure Across Endpoints
- **`/ask`** endpoint: Returns `QueryResponse` (question, answer, sources)
- **`/intelligence/analyze`** endpoint: Returns different structure with indicators, regime info
- **`/intelligence/hedge`** endpoint: Different structure again
- **Problem**: No unified response schema; formats are ad-hoc per endpoint

#### Issue #2: Hardcoded Format Instructions in Prompts
Location: `intelligence/macro_reasoner.py` lines 366-540
- Brief mode format block: 400+ chars of hardcoded instructions
- Detailed mode format block: 600+ chars of hardcoded instructions
- **Problem**: 
  - Brittle - changing format requires finding and updating multiple spots
  - Redundant across different prompt-building functions
  - No version control or standardization
  - Makes prompts harder to read and maintain

#### Issue #3: Scattered Response Building Logic
- **market_context.py**: 10+ section builders (yield curve, inflation, credit cycle, etc.)
- **macro_reasoner.py**: 
  - Event linking block building
  - Reasoning object formatting
  - COT (chain-of-thought) block construction
- **response_enhancer.py**: Post-hoc validation only, doesn't build responses
- **Problem**: Logic is distributed; no single source of truth for response structure

#### Issue #4: Unused Response Contract Files
- `docs/response_contract_v2.md` - **EMPTY**
- `intelligence/response_contract.py` - **EMPTY**
- **Problem**: Contract files exist but are not implemented; no schema definition

#### Issue #5: Manual Citation & Source Handling
- Citation format: `[Sx]` with normalization in `response_enhancer.py`
- Source handling scattered across:
  - RAG retrieval layer
  - Response building
  - Post-processing validation
- **Problem**: No standardized citation workflow; multiple places handle citations differently

#### Issue #6: Inconsistent Market Context Presentation
- Market context sections built in `market_context.py` use mixed formatting:
  - Some use emojis (`▲`, `▼`)
  - Some use plain text
  - Inconsistent line breaks and section headers
- **Problem**: User-facing output looks inconsistent

#### Issue #7: Complex Prompt Structure
- Multiple "blocks" in macro reasoning prompts:
  - Mechanics block (200+ chars)
  - Format block (400-600 chars)
  - COT block (300+ chars)
  - Market context integration block
  - Event linking block
  - Reasoning object block
- **Problem**: Prompts are hard to read, maintain, and version

#### Issue #8: No Reusable Response Components
- Response building is tightly coupled to specific models/endpoints
- No way to build a response incrementally or compose pieces
- **Problem**: Code duplication when building similar responses for different endpoints

---

## PART 2: RECOMMENDED IMPROVEMENTS

### 2.1 New Architecture Components to Add

#### Component #1: Response Schema Layer
**File**: `intelligence/response_schema.py`
**Purpose**: Define all response types as reusable dataclasses
**Contents**:
- `ResponseSection` - Base section with title, content, citations
- `DirectAnswer` - Structured answer with confidence
- `DataSnapshot` - Market data points with values and trends
- `CausalChain` - Trigger → Effects chain
- `MarketImpact` - Asset impacts (equities, rates, FX, commodities)
- `ScenarioSet` - Bull/Base/Bear scenarios with probabilities
- `PredictedEvent` - Near-term event with trigger, probability, invalidation
- `ResponseMetadata` - Quality metrics, confidence, sources, model info

#### Component #2: Response Builder System
**File**: `intelligence/response_builder.py`
**Purpose**: Construct responses from components
**Features**:
- `BriefResponseBuilder` - Builds brief-mode responses
- `DetailedResponseBuilder` - Builds detailed-mode responses  
- `HedgeResponseBuilder` - Builds hedge strategy responses
- Incremental building API
- Automatic validation after each section
- Consistent formatting and citations

#### Component #3: Prompt Template System
**File**: `intelligence/prompt_templates.py`
**Purpose**: Centralize all prompt text
**Contents**:
- Template classes for different prompt types
- Parameterizable format instructions
- Market context template blocks
- Mechanics blocks
- COT blocks
- Easy versioning and maintenance

#### Component #4: Writing Style Guidelines
**File**: `intelligence/writing_style.py`
**Purpose**: Enforce consistent writing style
**Features**:
- Tone guidelines (professional, analytical, data-driven)
- Emoji standardization (when to use, which ones)
- Formatting rules (bullet points, numbers, citations)
- Length guidelines (max chars per section)
- Technical terminology standards

#### Component #5: Response Normalizer
**File**: `intelligence/response_normalizer.py`
**Purpose**: Normalize responses from LLM before schema conversion
**Features**:
- Citation standardization `[Sx]` → parsed
- Probability detection and validation
- Number extraction and formatting
- Section extraction from free-form text
- Comment/note filtering

#### Component #6: Response Validator
**File**: `intelligence/response_validator.py`
**Purpose**: Comprehensive validation before returning response
**Features**:
- Schema validation
- Content quality checks
- Citation accuracy verification
- Probability sum validation (scenarios to 100%)
- Number plausibility checks
- Tone consistency

#### Component #7: Response Middleware
**File**: `intelligence/response_middleware.py`
**Purpose**: Normalize all responses before sending to client
**Features**:
- Convert all endpoint responses to unified format
- Add metadata (model, timestamp, quality score)
- Apply transformations (formatting, emoji standardization)
- Cache-friendly

### 2.2 Improvements to Existing Files

#### `intelligence/macro_reasoner.py`
- Move all prompt text to `prompt_templates.py`
- Replace hardcoded format blocks with template references
- Use `ResponseBuilder` instead of string construction
- Simplify `build_unified_response_prompt()` to use factory

#### `intelligence/response_enhancer.py`
- Refactor to work with response schema objects
- Add better integration with new validation layer
- Keep existing validation logic but improve

#### `intelligence/market_context.py`
- Keep existing section builders
- Add standardization layer to normalize output
- Create composable context builder

#### `api/app.py`
- Apply response middleware to all endpoints
- Update response models to support new schema
- Keep API contract unchanged for clients

#### `rag/query.py`
- Use response normalizer for RAG responses
- Apply new response builder for structured output
- Maintain backward compatibility

---

## PART 3: BENEFITS OF RENOVATION

### 3.1 Maintainability
- **Single source of truth** for response formats
- **Easier to update**: Change template once, affects all responses
- **Version control**: Track response format evolution
- **Reduced duplication**: Shared components across endpoints

### 3.2 Quality
- **Consistent output**: All responses follow same structure
- **Better validation**: Comprehensive checks before client sees response
- **Improved citations**: Standardized source tracking
- **Professional appearance**: Cohesive writing style

### 3.3 Flexibility
- **Easy new endpoint creation**: Use builders instead of building from scratch
- **Mode variations**: Brief/detailed/custom modes with shared core
- **Incremental building**: Build responses piece by piece
- **Easy A/B testing**: Swap builders or templates

### 3.4 Performance
- **Reusable validators**: Don't re-validate same patterns
- **Cached templates**: Pre-compiled prompt templates
- **Middleware optimization**: Batch normalizations
- **Better caching**: Standardized cache keys

---

## PART 4: IMPLEMENTATION STRATEGY

### Phase 1: Foundation (Components)
**Creates new components, no changes to existing code**
1. Create `intelligence/response_schema.py` - Define all dataclasses
2. Create `intelligence/writing_style.py` - Style guides
3. Create `intelligence/prompt_templates.py` - Centralize prompts
4. Create `intelligence/response_normalizer.py` - LLM output parsing
5. Create `intelligence/response_validator.py` - Validation logic
6. Create `intelligence/response_builder.py` - Builder classes
7. Create `intelligence/response_middleware.py` - Normalization layer

### Phase 2: Integration (Gradual Migration)
**Integrate new components without breaking existing**
1. Update `intelligence/macro_reasoner.py` to use templates
2. Update `intelligence/response_enhancer.py` for schema objects
3. Update `rag/query.py` to use response builder
4. Add response middleware to `api/app.py`
5. Update endpoint models to support new schema
6. Add migration layer for backward compatibility

### Phase 3: Optimization
**Performance tuning and final touches**
1. Cache prompt templates
2. Profile validation layer
3. Optimize response middleware
4. Clean up redundant code

### Phase 4: Documentation
**Document the new system**
1. Update `docs/response_contract_v2.md` with schema
2. Add examples in docstrings
3. Create migration guide for developers
4. Document writing style guidelines

---

## PART 5: MIGRATION CHECKLIST

### Code Quality
- [ ] All new code follows existing style
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Error handling with meaningful messages

### Testing
- [ ] Unit tests for schema validation
- [ ] Unit tests for builders
- [ ] Unit tests for normalizer
- [ ] Integration tests for middleware
- [ ] Backward compatibility tests

### Performance
- [ ] No latency regression on endpoints
- [ ] Validation doesn't exceed 10% of response time
- [ ] Memory usage reasonable for all components

### Documentation
- [ ] Response schema documented
- [ ] Builder API documented with examples
- [ ] Migration guide for developers
- [ ] Guidelines in WRITING_STYLE file

---

## PART 6: FILES TO CREATE/MODIFY

### NEW FILES (7 files)
```
intelligence/response_schema.py         (300-400 lines)
intelligence/writing_style.py           (200-300 lines)
intelligence/prompt_templates.py        (400-500 lines)
intelligence/response_normalizer.py     (200-300 lines)
intelligence/response_validator.py      (250-350 lines)
intelligence/response_builder.py        (400-500 lines)
intelligence/response_middleware.py     (150-250 lines)
```

### MODIFIED FILES (5 files)
```
intelligence/macro_reasoner.py          (cleanup/refactor: -200 lines)
intelligence/response_enhancer.py       (minor updates)
intelligence/market_context.py          (minor standardization)
rag/query.py                            (integrate builder)
api/app.py                              (add middleware)
```

### DOCUMENTATION FILES (2 files)
```
docs/response_contract_v2.md            (Response schema & examples)
docs/WRITING_STYLE_GUIDE.md             (Tone & formatting guidelines)
```

---

## PART 7: EXPECTED OUTCOMES

### Before Renovation
- Response format: Ad-hoc, inconsistent
- Response building: Scattered across files
- Prompts: Hardcoded in macro_reasoner.py
- Maintenance: Difficult, error-prone
- Quality: Inconsistent formatting
- Developer experience: Hard to add new response types

### After Renovation
- Response format: Unified schema, consistent
- Response building: Centralized builders
- Prompts: Versioned templates, easy to update
- Maintenance: Single source of truth
- Quality: Validation + normalization at every step
- Developer experience: Easy to create new response types

---

## NEXT STEPS

1. **Review this plan** - Confirm recommendations align with your goals
2. **Ask questions** - Clarify any aspects before implementation
3. **Start Phase 1** - Build foundation components
4. **Parallel Phase 2** - Integrate while foundation is being built
5. **Validate** - Test backward compatibility throughout
6. **Deploy** - Roll out incrementally, monitor quality metrics

---

**Status**: Ready for implementation approval  
**Estimated Effort**: 6-8 hours  
**Risk Level**: Low (backward compatible, phased approach)  
**Impact**: High (improves maintainability, quality, and developer experience)
