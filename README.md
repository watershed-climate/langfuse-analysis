# langfuse-analysis

Ad-hoc analysis of Langfuse trace data. Supports any Langfuse project — credentials and
target project are configured per-run.

## Setup

```bash
uv sync
```

Create a `.env` file with your Langfuse credentials:

```
LANGFUSE_HOST=https://us.cloud.langfuse.com
LANGFUSE_PROJECT=labs_agent

LANGFUSE_LABS_AGENT_PUBLIC_KEY=pk-lf-...
LANGFUSE_LABS_AGENT_SECRET_KEY=sk-lf-...
```

If you use 1Password, `.env.tpl` has `op://` references for all Watershed Langfuse
projects — run `./scripts/get-secrets` to populate `.env`.

## Projects

`LANGFUSE_PROJECT` picks which project a query targets. Every project has its own
cache file under `data/traces_{project}.db` and results under
`results/{query_name}/{project}/{timestamp}/`, so data from different projects never
mixes.

To add a project: set `LANGFUSE_{NAME}_PUBLIC_KEY` and `LANGFUSE_{NAME}_SECRET_KEY`
in `.env`, then pass `LANGFUSE_PROJECT={name}` when running the query. No code
changes required.

Projects currently wired up in `.env.tpl`:

| Project | Env var prefix |
| --- | --- |
| labs-agent | `LANGFUSE_LABS_AGENT_*` |
| production-graph | `LANGFUSE_PRODUCTION_GRAPH_*` |
| reporting-answers | `LANGFUSE_REPORTING_ANSWER_*` |
| ai-data-ingestion | `LANGFUSE_DATA_INGESTION_*` |
| ghg-sandbox | `LANGFUSE_SANDBOX_*` |
| test-project | `LANGFUSE_TEST_PROJECT_*` |
| ai-querying | `LANGFUSE_AI_QUERYING_*` |
| agentic-coding | `LANGFUSE_AGENTIC_CODING_*` |

## Running a query

```bash
# Default project (from $LANGFUSE_PROJECT in .env)
uv run python -m queries.prompt_type

# Target a different project one-off
LANGFUSE_PROJECT=reporting_answer uv run python -m queries.prompt_type
```

Results go to `results/{query}/{project}/{timestamp}/` with:
- `report.md` — summary tables and stats
- `data.json` — session-level data with Langfuse trace links

Traces are cached in `data/traces_{project}.db` (SQLite). The first run fetches
all historical traces from the API; subsequent runs fetch incrementally.

## Project structure

```
lib/
  config.py            # Per-project ProjectConfig (keys, host, cache path)
  langfuse_client.py   # API client, caching, pagination
  classify.py          # Trace classification (customer/internal/eval),
                       #   prompt type detection, rejection patterns
  cache.py             # SQLite cache layer (path per project)
  report.py            # Markdown table + file output helpers
queries/
  prompt_type.py       # Custom vs pre-canned prompt analysis
  failure_modes.py     # Session outcome classification
```

## Adding a new query

Create `queries/my_query.py`:

```python
from lib.config import get_project_config
from lib.langfuse_client import get_client, get_traces, group_by_session
from lib.report import save_results

config = get_project_config()
client = get_client(config)
traces = get_traces(client, config, from_timestamp=..., to_timestamp=...)
sessions = group_by_session(traces)

# ... your analysis ...

save_results("my_query", report_md, data_dict, project=config.name)
```

Run with `uv run python -m queries.my_query`.
