"""Langfuse API client with rate-limit-aware pagination and SQLite caching.

First run fetches from the API and caches to SQLite as it goes.
Subsequent runs only fetch new traces since the last cached timestamp.

Each project has its own cache (see lib.config.ProjectConfig.cache_path)
so traces from different Langfuse projects are never mixed.
"""

import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from langfuse import Langfuse

from lib.cache import (
    cache_stats,
    get_cached_traces,
    get_db,
    get_latest_timestamp,
    log_fetch,
    upsert_traces,
)
from lib.config import ProjectConfig

# Rate limiting: Langfuse cloud allows ~100 req/min.
# With page size 1000, we make far fewer requests.
DELAY_BETWEEN_REQUESTS = 0.1
PAGE_SIZE = 100
RATE_LIMIT_BACKOFF_BASE = 10
RATE_LIMIT_BACKOFF_MAX = 120
MAX_RETRIES = 5


def get_client(config: ProjectConfig) -> Langfuse:
    """Create a Langfuse client for the given project config."""
    return Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        host=config.host,
        timeout=120,
    )


def get_traces(
    client: Langfuse,
    config: ProjectConfig,
    *,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
) -> list[dict]:
    """Get traces for a time range, using per-project cache + incremental API fetch.

    Caches each page as it's fetched so progress is never lost.
    """
    conn = get_db(config.cache_path)
    stats = cache_stats(conn)
    print(f"  Project '{config.name}' cache ({config.cache_path}): "
          f"{stats['total']} traces ({stats['earliest'] or 'empty'} to {stats['latest'] or 'empty'})")

    now = datetime.now(timezone.utc)
    to_ts = to_timestamp or now
    from_ts = from_timestamp

    latest_cached = get_latest_timestamp(conn)
    if latest_cached:
        api_from = datetime.fromisoformat(latest_cached.replace("Z", "+00:00"))
        print(f"  Incremental fetch since {api_from.date()}...")
    else:
        api_from = from_ts
        print(f"  Cold start — fetching from {api_from.date() if api_from else 'beginning'}...")

    # Fetch from API, caching each page as we go
    total_new = _fetch_and_cache(client, conn, from_timestamp=api_from, to_timestamp=to_ts)
    if total_new:
        log_fetch(conn, project=config.name, from_ts=str(api_from or ""), to_ts=to_ts.isoformat(), count=total_new)
        print(f"  Cached {total_new} new traces")
    else:
        print(f"  Cache is up to date")

    # Serve from cache
    from_str = from_ts.isoformat() if from_ts else None
    to_str = to_ts.isoformat() if to_ts else None
    traces = get_cached_traces(conn, from_ts=from_str, to_ts=to_str)
    conn.close()

    print(f"  Returning {len(traces)} traces")
    return traces


def group_by_session(traces: list[dict]) -> dict[str, list[dict]]:
    """Group traces by session_id, sorted by timestamp within each session."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        sid = t.get("session_id")
        if sid:
            sessions[sid].append(t)
    for trace_list in sessions.values():
        trace_list.sort(key=lambda t: t.get("timestamp") or "")
    return dict(sessions)


def trace_url(trace_id: str, host: str | None = None) -> str:
    """Generate a Langfuse dashboard link for a trace."""
    h = (host or os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")).rstrip("/")
    return f"{h}/trace/{trace_id}"


def _fetch_and_cache(
    client: Langfuse,
    conn,
    *,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
) -> int:
    """Fetch traces from API page by page, caching each page immediately."""
    page = 1
    total_new = 0
    kwargs: dict = {"limit": PAGE_SIZE}
    if from_timestamp:
        kwargs["from_timestamp"] = from_timestamp
    if to_timestamp:
        kwargs["to_timestamp"] = to_timestamp

    while True:
        print(f"    API page {page}...", end=" ", flush=True)
        result = _api_call(client.api.trace.list, page=page, **kwargs)
        if not result or not result.data:
            print("done")
            break
        batch = [_trace_to_dict(t) for t in result.data]
        inserted = upsert_traces(conn, batch)
        total_new += inserted
        print(f"{len(result.data)} traces ({inserted} new, {total_new} total new)")
        if len(result.data) < PAGE_SIZE:
            break
        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return total_new


def _trace_to_dict(t) -> dict:
    """Convert a Langfuse trace API object to a plain dict."""
    return {
        "id": t.id,
        "name": t.name,
        "session_id": t.session_id,
        "user_id": t.user_id,
        "tags": t.tags or [],
        "environment": getattr(t, "environment", None),
        "metadata": t.metadata or {},
        "input": t.input,
        "output": t.output,
        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
    }


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "too many" in msg


def _api_call(fn, *args, **kwargs):
    """Call an API function with exponential backoff on rate limits."""
    backoff = RATE_LIMIT_BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limited(e):
                wait = min(backoff, RATE_LIMIT_BACKOFF_MAX)
                print(f"\n    Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                backoff *= 2
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            raise
    return None
