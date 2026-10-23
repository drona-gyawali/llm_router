# LLM Router

An experimental internal LLM gateway that routes requests across multiple AI providers with multi-layer guardrails, semantic routing, tenant management, and cost tracking.

---

## Bring Your Own Model

This project is fully config-driven. The example models in `models.yaml` (Claude, GPT-4, Gemini) are there for reference — **you are not locked into any of them.**

To add or swap any model, edit `models.yaml` only. No code changes required.

```yaml
models:
  - name: "my-custom-model"
    provider: "openai"          # anthropic | openai | google
    model_id: "gpt-4o"
    base_url: "https://api.openai.com/v1"
    api_key_env_var: "MY_API_KEY"
    api_schema:
      endpoint_path: "/chat/completions"
      auth_header: "Authorization"
      auth_scheme: "Bearer"
      ...
```

Same for routing — assign any category to any model in `models.yaml`:

```yaml
routing:
  categories:
    coding:    "my-custom-model"   # ← just change this
    reasoning: "my-custom-model"
    general:   "my-custom-model"
```

Restart the server. Done.

---

## Architecture

![LLM Router Architecture](diagram/architecture.png)

---

## Important Note on Datasets

> **The datasets included in this project are biased and intended for development and testing only. If you are running this in production, bring your own high-quality labeled data.**

This project depends heavily on semantic context. The quality of your guardrails and routing directly reflects the quality of your data. Garbage in, garbage out.

There are two types of datasets:

### `data/` — Guardrail Datasets

Used to train the semantic guardrail layer.

```
data/
├── references_raw/    ← positive reference examples per category
│                        (what harmful content looks like)
├── references/        ← auto-generated .npy embeddings from references_raw
├── raw/               ← labeled eval data (label 1 = harmful, 0 = safe)
└── thresholds.json    ← auto-generated optimal thresholds from benchmark
```

Bring your own `references_raw/` and `raw/` data for each guardrail category. The better your examples, the more accurate your guardrails.

### `data/routing/` — Routing Datasets

Used to train the semantic routing layer.

```
data/routing/
├── references_raw/    ← positive examples per routing category
│                        (what coding/reasoning/creative etc. look like)
├── references/        ← auto-generated .npy embeddings
├── raw/               ← labeled eval data per category
└── thresholds.json    ← auto-generated optimal thresholds
```

Bring your own examples that reflect the actual requests your engineers send. The routing is only as smart as the examples you provide.

### Rebuilding After Bringing Your Own Data

```bash
# Guardrails
python scripts/build_reference.py
python scripts/benchmark_threshold.py

# Routing
python scripts/build_reference.py \
    --source data/routing/references_raw \
    --output data/routing/references

python scripts/benchmark_threshold.py \
    --raw    data/routing/raw \
    --refs   data/routing/references \
    --output data/routing/thresholds.json
```

---

## Routing Categories

The semantic router classifies requests into: `coding`, `reasoning`, `creative`, `document`, `data`, `general`. Each maps to a model in `models.yaml` — change the mapping anytime without touching code.

## Guardrail Categories

The semantic guardrail covers: `bad_words`, `sexual`, `sexual_harm`, `violent_harmful`, `medical_pii`, `prompt_injection`, `sql_injection`. Thresholds are auto-tuned when you run `benchmark_threshold.py` against your own data.

---

## Getting Started

### Prerequisites

```
Python 3.9+
PostgreSQL
Redis
```

### Install

```bash
git clone <repo>
cd llm_router
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
DATABASE_URL=postgresql+asyncpg://postgres:secretpassword@localhost:5432/mydatabase
WEB_PORT=8080
REDIS_HOST=localhost
REDIS_PORT=6379

## BRING YOU OWN MODEL for more details refers to config/models.yaml
GROQ_API_KEY=xxx
GOOGLE_API_KEY=xxxx

```

### Run Migrations

```bash
alembic upgrade head
```

### Start Server

```bash
uvicorn main:app --reload
```

---

## API Reference

### Create Tenant

```bash
POST /auth/tenants
Content-Type: application/json

{
  "name": "my-team",
  "monthly_budget": 500.0
}
```

### Create API Key

```bash
POST /auth/tenants/{tenant_id}/keys
Content-Type: application/json

{
  "name": "production-key"
}
```

> The `raw_key` is returned **once only**. Save it immediately — it is never stored.

### Send Chat Request

```bash
POST /chat
X-API-KEY: gw_live_...
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "write a Python function to sort a list"}
  ],
  "stream": false,
  "max_tokens": 1024,
  "temperature": 0.7
}
```

### Response

```json
{
  "id": "abc123",
  "content": "...",
  "model_used": "claude-3-5-sonnet-20241022",
  "category": "coding",
  "routing_rule": "SEMANTIC:CODING",
  "input_tokens": 24,
  "output_tokens": 187
}
```

### Streaming

```bash
POST /chat
{
  "messages": [...],
  "stream": true
}
```

Returns SSE stream. Response headers include `X-Model-Used`, `X-Routing-Rule`, `X-Category`.

---

## Interactive Docs

```
http://localhost:8000/docs
```