#!/usr/bin/env python3
"""Prompt type analysis: custom prompt box vs pre-canned suggestions.

Strategy for accuracy:
1. Fetch a wide historical window (90 days) to identify each user's first-ever session
2. Only analyze the last 30 days of customer (production) traces
3. Exclude any session that is a user's first-ever session (onboarding noise)
4. Use Langfuse tags/environment to identify real customers vs internal/eval traces

Usage:
    uv run python -m queries.prompt_type
    INCLUDE_INTERNAL=1 uv run python -m queries.prompt_type   # include internal users
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from lib.classify import (
    TraceCategory,
    classify_prompt,
    classify_session_outcome,
    classify_trace,
    extract_input_text,
    strip_file_upload_prefix,
)
from lib.langfuse_client import get_client, get_traces, group_by_session, trace_url
from lib.report import df_to_markdown_table, markdown_table, save_results

SUGGESTION_LABELS = {
    "clean_data": "Clean my raw data",
    "emissions_hotspots": "Show me emissions hotspots",
    "yoy_analysis": "Analyze drivers of YoY change",
}

PROMPT_TYPE_ORDER = ["custom", "pre_canned", "system_auto", "greeting_test"]
PROMPT_TYPE_LABELS = {
    "custom": "Custom (user-typed)",
    "pre_canned": "Pre-canned suggestion",
    "system_auto": "System/automated",
    "greeting_test": "Greeting/test",
}
PROMPT_TYPE_SHORT = {
    "custom": "Custom",
    "pre_canned": "Pre-canned",
    "system_auto": "System/auto",
    "greeting_test": "Greeting/test",
}

ANALYSIS_DAYS = 30
HISTORY_DAYS = 45


def main():
    load_dotenv()
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    include_internal = os.getenv("INCLUDE_INTERNAL", "").lower() in ("1", "true", "yes")

    now = datetime.now(timezone.utc)
    analysis_start = now - timedelta(days=ANALYSIS_DAYS)
    history_start = now - timedelta(days=HISTORY_DAYS)

    client = get_client(
        public_key_env="LANGFUSE_LABS_AGENT_PUBLIC_KEY",
        secret_key_env="LANGFUSE_LABS_AGENT_SECRET_KEY",
    )

    # Step 1: Fetch the full history window to find first-ever sessions
    print(f"Fetching historical traces ({history_start.date()} to {now.date()}) "
          f"to identify first sessions...")
    all_traces = get_traces(client, from_timestamp=history_start, to_timestamp=now)
    all_sessions = group_by_session(all_traces)
    print(f"  {len(all_traces)} traces across {len(all_sessions)} sessions\n")

    # Step 2: Build a map of each user's first-ever session ID (across full history)
    user_first_session_id: dict[str, tuple[str, str]] = {}
    for sid, sess_traces in all_sessions.items():
        if not sess_traces:
            continue
        first = sess_traces[0]
        uid = first.get("user_id") or "anonymous"
        ts = first.get("timestamp") or ""
        if uid not in user_first_session_id:
            user_first_session_id[uid] = (ts, sid)
        elif ts < user_first_session_id[uid][0]:
            user_first_session_id[uid] = (ts, sid)
    first_session_ids = {sid for _, sid in user_first_session_id.values()}

    # Step 3: Filter to analysis window
    analysis_traces = [
        t for t in all_traces
        if t.get("timestamp") and t["timestamp"] >= analysis_start.isoformat()
    ]
    sessions = group_by_session(analysis_traces)
    print(f"Analysis window ({analysis_start.date()} to {now.date()}): "
          f"{len(analysis_traces)} traces, {len(sessions)} sessions\n")

    # Step 4: Build session-level records
    all_records = []
    for sid, sess_traces in sessions.items():
        if not sess_traces:
            continue
        first = sess_traces[0]
        user_id = first.get("user_id")
        meta = first.get("metadata") or {}
        category = classify_trace(first)
        input_text = extract_input_text(first.get("input"))
        stripped = strip_file_upload_prefix(input_text)
        prompt_class, suggestion_name = classify_prompt(input_text)

        session_analysis = classify_session_outcome(sess_traces)

        all_records.append({
            "session_id": sid,
            "user_id": user_id,
            "user_name": meta.get("userName"),
            "category": category.value,
            "org_id": meta.get("orgId", ""),
            "org_name": meta.get("orgName", ""),
            "is_demo_org": meta.get("isDemoOrg"),
            "environment": first.get("environment"),
            "tags": first.get("tags", []),
            "timestamp": first.get("timestamp", ""),
            "num_turns": len(sess_traces),
            "first_input": stripped[:500],
            "prompt_class": prompt_class,
            "suggestion_name": suggestion_name,
            "outcome": session_analysis.outcome.value,
            "total_rejections": session_analysis.total_rejections,
            "rejection_turns": session_analysis.rejection_turns,
            "has_positive_close": session_analysis.has_positive_close,
            "rejection_detected": session_analysis.total_rejections > 0,
            "trace_id": first["id"],
            "trace_url": trace_url(first["id"], host),
            "is_first_session": sid in first_session_ids,
        })

    df_all = pd.DataFrame(all_records)
    if df_all.empty:
        print("No sessions found in analysis window.")
        return

    # Step 5: Filter by category
    if include_internal:
        df_cat = df_all
    else:
        df_cat = df_all[df_all["category"] == TraceCategory.CUSTOMER.value]

    # Step 6: Exclude first-ever sessions
    df_filtered = df_cat[~df_cat["is_first_session"]]

    # Build report
    lines: list[str] = []
    def out(s=""):
        lines.append(s)

    out("# Prompt Type Analysis: Custom vs Pre-Canned Suggestions")
    out(f"**Generated**: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    out(f"**Analysis window**: {analysis_start.date()} to {now.date()} ({ANALYSIS_DAYS} days)")
    out(f"**History window for first-session detection**: {history_start.date()} to {now.date()} ({HISTORY_DAYS} days)")
    out(f"**Data**: {len(analysis_traces)} traces across {len(sessions)} sessions")
    out()
    filter_label = "All traces (internal included)" if include_internal else "Customers only"
    out(f"**Filter mode**: {filter_label}")
    out()
    out("**Filters applied**:")
    out(f"1. Excluded non-customer traces: {len(df_all) - len(df_cat)} sessions removed")
    first_excluded = len(df_cat) - len(df_filtered)
    out(f"2. Excluded first-ever session per user (using {HISTORY_DAYS}-day history): {first_excluded} sessions removed")
    out(f"3. **Remaining sessions for analysis: {len(df_filtered)}**")
    out()
    out("---")
    out()

    df = df_filtered if len(df_filtered) > 0 else df_cat
    label_note = "" if len(df_filtered) > 0 else " (NOTE: includes first sessions since no repeat sessions exist)"

    # Parse timestamps for time-based aggregations
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True, errors="coerce")

    # Section 1: Overall summary
    out("## 1. Overall Summary")
    out()
    if len(df_filtered) == 0:
        out("**No sessions remain after filtering.** All customer sessions in this window are first-time.")
        out()
        out("### Unfiltered Customer View")
        out()

    total = len(df)
    prompt_counts = df["prompt_class"].value_counts()
    rows = []
    for ptype in PROMPT_TYPE_ORDER:
        cnt = int(prompt_counts.get(ptype, 0))
        pct = f"{100 * cnt / total:.1f}%" if total else "0%"
        rows.append([PROMPT_TYPE_LABELS[ptype], cnt, pct])
    rows.append(["**Total**", f"**{total}**", "**100%**"])
    out(markdown_table(["Prompt Type", "Sessions", "% of Total"], rows))
    out()

    # Section 2: By user
    out(f"## 2. By Individual Customer{label_note}")
    out()
    user_grp = df.groupby(df["user_id"].fillna("no_user_id"))
    user_rows = []
    for uid, udf in sorted(user_grp, key=lambda x: -len(x[1])):
        uc = udf["prompt_class"].value_counts()
        t = len(udf)
        custom_n = int(uc.get("custom", 0))
        cpct = f"{100 * custom_n / t:.0f}%" if t else "0%"
        org_name = udf.iloc[0]["org_name"] or str(udf.iloc[0]["org_id"])[:16] or "-"
        other = int(uc.get("system_auto", 0)) + int(uc.get("greeting_test", 0))
        user_rows.append([
            str(uid)[:26], str(org_name)[:20],
            custom_n, int(uc.get("pre_canned", 0)), other, t, cpct,
        ])
    out(markdown_table(["User ID", "Org", "Custom", "Pre-canned", "Other", "Total", "Custom %"], user_rows))
    out()

    # Section 3: Pre-canned breakdown
    out("## 3. Which Pre-Canned Suggestions Are Used?")
    out()
    precanned = df[df["prompt_class"] == "pre_canned"]
    sugg_counts = precanned["suggestion_name"].value_counts()
    rows = []
    for name in ["clean_data", "emissions_hotspots", "yoy_analysis"]:
        rows.append([SUGGESTION_LABELS[name], int(sugg_counts.get(name, 0))])
    out(markdown_table(["Suggestion Chip", "Times Used"], rows))
    out()

    # Section 4: Rejection analysis (full-session scan)
    out("## 4. Rejection / Redirection (Any Turn)")
    out()
    multi_turn = df[df["num_turns"] >= 2]
    rejections = multi_turn[multi_turn["rejection_detected"]]
    out(f"- Sessions with 2+ turns: **{len(multi_turn)}**")
    out(f"- Sessions with rejection detected: **{len(rejections)}**")
    if len(multi_turn) > 0:
        rate = 100 * len(rejections) / len(multi_turn)
        out(f"- Rejection rate: **{rate:.1f}%**")
    out()

    out("### Rejection Rate by Prompt Type")
    out()
    rows = []
    for ptype in PROMPT_TYPE_ORDER:
        mt = multi_turn[multi_turn["prompt_class"] == ptype]
        rej = mt[mt["rejection_detected"]]
        r = f"{100 * len(rej) / len(mt):.0f}%" if len(mt) > 0 else "-"
        rows.append([PROMPT_TYPE_SHORT[ptype], len(mt), len(rej), r])
    out(markdown_table(
        ["Prompt Type", "Multi-turn Sessions",
         "Rejections", "Rejection Rate"],
        rows,
    ))
    out()

    if len(rejections) > 0:
        out("### Sessions With Rejections")
        out()
        rows = []
        for _, r in rejections.sort_values("timestamp").iterrows():
            ts = str(r["timestamp"])[:10] or "?"
            uid = str(r["user_id"] or "?")[:20]
            sugg = (
                SUGGESTION_LABELS.get(r["suggestion_name"], "-")
                if r["suggestion_name"] else "-"
            )
            turns_str = ", ".join(str(t) for t in r["rejection_turns"])
            rows.append([
                ts, uid, r["prompt_class"], sugg,
                r["total_rejections"], turns_str,
            ])
        out(markdown_table(
            ["Date", "User", "Prompt Type", "Suggestion",
             "Rejections", "Turns"],
            rows,
        ))
        out()

    # Section 4b: Session outcomes
    out("## 4b. Session Outcomes")
    out()
    outcome_counts = df["outcome"].value_counts()
    rows = []
    for outcome in ["completed", "abandoned", "redirected"]:
        cnt = int(outcome_counts.get(outcome, 0))
        pct = f"{100 * cnt / total:.1f}%" if total else "0%"
        rows.append([outcome.capitalize(), cnt, pct])
    out(markdown_table(
        ["Outcome", "Sessions", "% of Total"], rows,
    ))
    out()

    out("### Outcome by Prompt Type")
    out()
    outcome_by_type = (
        df.groupby(["prompt_class", "outcome"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["completed", "abandoned", "redirected"]:
        if col not in outcome_by_type.columns:
            outcome_by_type[col] = 0
    outcome_by_type = outcome_by_type[
        ["completed", "abandoned", "redirected"]
    ]
    outcome_by_type.index = outcome_by_type.index.map(
        lambda x: PROMPT_TYPE_SHORT.get(x, x)
    )
    outcome_by_type.index.name = "Prompt Type"
    outcome_by_type.columns = [
        c.capitalize() for c in outcome_by_type.columns
    ]
    out(df_to_markdown_table(outcome_by_type))
    out()

    # Section 5: Engagement
    out("## 5. Engagement: Custom vs Pre-canned")
    out()
    for ptype, label in [("custom", "Custom prompt"), ("pre_canned", "Pre-canned suggestion")]:
        turns = df.loc[df["prompt_class"] == ptype, "num_turns"]
        if len(turns) > 0:
            out(f"**{label} sessions** ({len(turns)} sessions):")
            out(f"- Avg turns: {turns.mean():.1f}")
            out(f"- Median turns: {int(turns.median())}")
            out(f"- Max turns: {int(turns.max())}")
            out()

    # Section 6: Week over week
    out("## 6. Week-over-Week Trends")
    out()
    df_with_ts = df.dropna(subset=["ts"])
    if len(df_with_ts) > 0:
        df_with_ts = df_with_ts.copy()
        iso_cal = df_with_ts["ts"].dt.isocalendar()
        df_with_ts["week_label"] = iso_cal["year"].astype(str) + "-W" + iso_cal["week"].astype(str).str.zfill(2)
        weekly = df_with_ts.groupby(["week_label", "prompt_class"]).size().unstack(fill_value=0)
        for col in PROMPT_TYPE_ORDER:
            if col not in weekly.columns:
                weekly[col] = 0
        weekly = weekly[PROMPT_TYPE_ORDER]
        weekly["Total"] = weekly.sum(axis=1)
        weekly["Custom %"] = (100 * weekly["custom"] / weekly["Total"]).round(0).astype(int).astype(str) + "%"
        weekly.columns = ["Custom", "Pre-canned", "System", "Greeting", "Total", "Custom %"]
        weekly.index.name = "Week"
        out(df_to_markdown_table(weekly))
    out()

    # Section 7: Custom prompts list
    out("## 7. All Custom Prompts (Customers)")
    out()
    custom_df = df[df["prompt_class"] == "custom"].sort_values("timestamp")
    seen: set[str] = set()
    for _, r in custom_df.iterrows():
        inp = str(r["first_input"]).replace("\n", " ").strip()[:300]
        if inp not in seen:
            seen.add(inp)
            ts = str(r["timestamp"])[:10] if r["timestamp"] else "?"
            org = r["org_name"] or "?"
            out(f"- [{ts}] {org}: {inp}")
    out()

    # Section 8: Appendix
    out("---")
    out()
    out("## Appendix: Unfiltered Totals")
    out()
    cat_counts = df_all["category"].value_counts()
    out("### Trace categories (before filtering)")
    out()
    out(markdown_table(
        ["Category", "Sessions"],
        [[cat, int(cat_counts.get(cat, 0))] for cat in ["customer", "internal", "eval", "unknown"]],
    ))
    out()

    first_sessions = df_cat[df_cat["is_first_session"]]
    if len(first_sessions) > 0:
        out("### Excluded first sessions")
        out()
        rows = []
        for _, r in first_sessions.sort_values("timestamp").iterrows():
            ts = str(r["timestamp"])[:10] if r["timestamp"] else "?"
            org = r["org_name"] or "-"
            rows.append([
                ts, str(r["user_id"] or "?")[:22], str(org)[:20],
                r["prompt_class"], str(r["first_input"]).replace("\n", " ")[:80],
            ])
        out(markdown_table(["Date", "User", "Org", "Type", "First Input"], rows))
        out()

    out(markdown_table(
        ["Scope", "Sessions"],
        [
            ["All sessions (analysis window)", len(df_all)],
            ["After category filter", len(df_cat)],
            ["After excluding first sessions", len(df_filtered)],
        ],
    ))
    out()

    report_md = "\n".join(lines)

    data = {
        "generated": now.isoformat(),
        "analysis_window": {"from": analysis_start.isoformat(), "to": now.isoformat()},
        "history_window": {"from": history_start.isoformat(), "to": now.isoformat()},
        "include_internal": include_internal,
        "filters": {
            "total_sessions": len(df_all),
            "excluded_non_customer": len(df_all) - len(df_cat),
            "excluded_first_sessions": len(df_cat) - len(df_filtered),
            "remaining": len(df_filtered),
        },
        "category_breakdown": cat_counts.to_dict(),
        "summary": prompt_counts.to_dict(),
        "sessions": df[[
            "session_id", "user_id", "user_name", "category",
            "org_id", "org_name", "environment",
            "prompt_class", "suggestion_name",
            "outcome", "rejection_detected",
            "total_rejections", "rejection_turns",
            "has_positive_close",
            "num_turns", "first_input",
            "trace_url", "trace_id", "is_first_session",
        ]].rename(columns={
            "prompt_class": "prompt_type",
            "suggestion_name": "suggestion",
            "rejection_detected": "rejection",
        }).to_dict(orient="records"),
    }

    save_results("prompt_type", report_md, data)


if __name__ == "__main__":
    main()
