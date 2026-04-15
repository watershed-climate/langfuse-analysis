# langfuse-analysis

Ad-hoc analysis of Langfuse trace data from the Labs AI agent.

## Setup

```bash
uv sync
./scripts/get-secrets   # requires 1Password CLI
```

This resolves secrets from `.env.tpl` into `.env`. If you don't use 1Password, copy `.env.example` and fill in the keys manually.

## Running a query

```bash
uv run python -m queries.prompt_type
```

Results go to `results/prompt_type/{timestamp}/` with:
- `report.md` — summary tables and stats
- `data.json` — session-level data with Langfuse trace links

Traces are cached in `data/traces.db` (SQLite). The first run fetches all historical traces from the API; subsequent runs fetch incrementally.

## Project structure

```
lib/
  langfuse_client.py   # API client, caching, pagination
  classify.py          # Trace classification (customer/internal/eval),
                       #   prompt type detection, rejection patterns
  cache.py             # SQLite cache layer
  report.py            # Markdown table + file output helpers
queries/
  prompt_type.py       # Custom vs pre-canned prompt analysis
```

## Adding a new query

Create `queries/my_query.py`:

```python
from lib.langfuse_client import get_client, get_traces, group_by_session
from lib.classify import classify_trace, TraceCategory
from lib.report import save_results

client = get_client("LANGFUSE_LABS_AGENT_PUBLIC_KEY", "LANGFUSE_LABS_AGENT_SECRET_KEY")
traces = get_traces(client, from_timestamp=..., to_timestamp=...)
sessions = group_by_session(traces)

# ... your analysis ...

save_results("my_query", report_md, data_dict)
```

Run with `uv run python -m queries.my_query`.
