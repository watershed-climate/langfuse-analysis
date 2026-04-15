"""SQLite cache for Langfuse traces.

Stores every trace we've ever fetched so subsequent queries are instant.
Only new traces (by timestamp) are fetched from the API.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    name TEXT,
    session_id TEXT,
    user_id TEXT,
    tags TEXT,
    environment TEXT,
    metadata TEXT,
    input TEXT,
    output TEXT,
    timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT,
    from_ts TEXT,
    to_ts TEXT,
    traces_fetched INTEGER,
    fetched_at TEXT
);
"""

_DB_PATH = "data/traces.db"


def get_db(path: str = _DB_PATH) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert_traces(conn: sqlite3.Connection, traces: list[dict]) -> int:
    """Insert traces into cache, skipping duplicates. Returns count of new traces."""
    new = 0
    for t in traces:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO traces
                   (id, name, session_id, user_id, tags, environment,
                    metadata, input, output, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t["id"],
                    t.get("name"),
                    t.get("session_id"),
                    t.get("user_id"),
                    json.dumps(t.get("tags", [])),
                    t.get("environment"),
                    json.dumps(t.get("metadata", {}), default=str),
                    json.dumps(t.get("input"), default=str),
                    json.dumps(t.get("output"), default=str),
                    t.get("timestamp"),
                ),
            )
            if conn.total_changes:
                new += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return new


def get_cached_traces(
    conn: sqlite3.Connection,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> list[dict]:
    """Read traces from cache, optionally filtered by timestamp range."""
    clauses = []
    params: list = []
    if from_ts:
        clauses.append("timestamp >= ?")
        params.append(from_ts)
    if to_ts:
        clauses.append("timestamp <= ?")
        params.append(to_ts)

    sql = "SELECT * FROM traces"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY timestamp"

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_latest_timestamp(conn: sqlite3.Connection) -> str | None:
    """Get the most recent trace timestamp in the cache."""
    row = conn.execute("SELECT MAX(timestamp) FROM traces").fetchone()
    return row[0] if row and row[0] else None


def log_fetch(conn: sqlite3.Connection, project: str, from_ts: str, to_ts: str, count: int):
    conn.execute(
        "INSERT INTO fetch_log (project, from_ts, to_ts, traces_fetched, fetched_at) VALUES (?, ?, ?, ?, ?)",
        (project, from_ts, to_ts, count, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def cache_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    earliest = conn.execute("SELECT MIN(timestamp) FROM traces").fetchone()[0]
    latest = conn.execute("SELECT MAX(timestamp) FROM traces").fetchone()[0]
    return {"total": total, "earliest": earliest, "latest": latest}


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "session_id": row["session_id"],
        "user_id": row["user_id"],
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "environment": row["environment"],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        "input": json.loads(row["input"]) if row["input"] else None,
        "output": json.loads(row["output"]) if row["output"] else None,
        "timestamp": row["timestamp"],
    }
