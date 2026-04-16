"""Report generation utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """Generate a markdown table from headers and rows."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table.

    Includes the index as the first column(s) unless it is a
    default RangeIndex.
    """
    import pandas

    if not isinstance(df.index, pandas.RangeIndex):
        df = df.reset_index()
    headers = [str(c) for c in df.columns]
    rows = [
        [str(v) for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return markdown_table(headers, rows)


def save_results(
    query_name: str,
    report_md: str,
    data_dict: dict,
    project: str | None = None,
    results_dir: str = "results/",
) -> Path:
    """Save timestamped report.md and data.json.

    Path: results/{query_name}/{project}/{timestamp}/ (if project given)
          results/{query_name}/{timestamp}/            (legacy, no project)
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    base = Path(results_dir) / query_name
    out = base / project / ts if project else base / ts
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.md").write_text(report_md)
    (out / "data.json").write_text(json.dumps(data_dict, indent=2, default=str))

    print(f"Results saved to {out}/")
    return out
