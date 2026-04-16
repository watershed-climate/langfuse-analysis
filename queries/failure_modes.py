#!/usr/bin/env python3
"""Failure mode analysis: classify every customer session into an outcome category.

Classifies sessions into: clean_success, success_after_correction, file_then_continued,
agent_answered_question, stalled_awaiting_user, single_turn_abandoned, api_auth_failure, other.

Usage:
    uv run python -m queries.failure_modes
    INCLUDE_INTERNAL=1 uv run python -m queries.failure_modes   # include internal users
"""

import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from lib.classify import (
    TraceCategory,
    classify_prompt,
    classify_trace,
    detect_rejection,
    extract_input_text,
    strip_file_upload_prefix,
)
from lib.config import get_project_config
from lib.langfuse_client import get_client, get_traces, group_by_session, trace_url
from lib.report import df_to_markdown_table, markdown_table, save_results

ANALYSIS_DAYS = 30
HISTORY_DAYS = 45

# --- Detection helpers ---

_FILE_DELIVERED_RE = re.compile(
    r"output/|\.xlsx|\.csv|\bworkbook\b|file\s.*saved", re.IGNORECASE
)
_AUTH_FAILURE_RE = re.compile(r"401.*Unauthorized|Unauthorized.*401", re.IGNORECASE)
_QUESTION_END_RE = re.compile(r"\?\s*$")
_MARKDOWN_TABLE_RE = re.compile(r"\|.*\|.*\|")

OUTCOME_ORDER = [
    "clean_success",
    "success_after_correction",
    "file_then_continued",
    "agent_answered_question",
    "stalled_awaiting_user",
    "single_turn_abandoned",
    "api_auth_failure",
    "other",
]

OUTCOME_LABELS = {
    "clean_success": "Clean success",
    "success_after_correction": "Success after correction",
    "file_then_continued": "File delivered, then follow-ups",
    "agent_answered_question": "Agent answered question (no file)",
    "stalled_awaiting_user": "Stalled awaiting user",
    "single_turn_abandoned": "Single turn abandoned",
    "api_auth_failure": "API auth failure",
    "other": "Other",
}

OUTCOME_DESCRIPTIONS = {
    "clean_success": "File delivered, no corrections",
    "success_after_correction": "File delivered after user corrected agent",
    "file_then_continued": "File delivered, user asked follow-ups",
    "agent_answered_question": "No file delivery, substantive analytical answer",
    "stalled_awaiting_user": "Agent asked question, user never came back",
    "single_turn_abandoned": "User sent 1 message, never followed up",
    "api_auth_failure": "401 errors — infra problem",
    "other": "Doesn't fit above categories",
}


def _output_text(trace: dict) -> str:
    """Extract output text from a trace."""
    out = trace.get("output")
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        content = out.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", str(item)))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(out)
    return str(out)


def _has_file_delivery(output: str) -> bool:
    return bool(_FILE_DELIVERED_RE.search(output))


def _has_auth_failure(output: str) -> bool:
    return bool(_AUTH_FAILURE_RE.search(output))


def _is_substantive_answer(output: str) -> bool:
    """No file delivery, but agent gave a real analytical answer."""
    if len(output) > 500:
        return True
    if _MARKDOWN_TABLE_RE.search(output):
        return True
    return False


def _ends_with_question(output: str) -> bool:
    return bool(_QUESTION_END_RE.search(output.strip()))


def classify_failure_mode(traces: list[dict]) -> dict:
    """Classify a session into a failure mode category.

    Returns dict with outcome, details about the classification,
    and supporting evidence.
    """
    num_turns = len(traces)
    has_auth = False
    has_file = False
    file_turn = None
    has_correction = False
    correction_details: list[dict] = []
    last_output = ""

    for i, t in enumerate(traces):
        output = _output_text(t)
        inp = extract_input_text(t.get("input"))

        # Check auth failure
        if _has_auth_failure(output):
            has_auth = True

        # Check file delivery
        if _has_file_delivery(output):
            if not has_file:
                file_turn = i
            has_file = True

        # Check user correction (turn 2+)
        if i > 0 and inp:
            is_rej, context = detect_rejection(inp)
            if is_rej:
                has_correction = True
                prev_output = _output_text(traces[i - 1]) if i > 0 else ""
                correction_details.append({
                    "turn": i,
                    "agent_output_before": prev_output[:500],
                    "user_correction": inp[:500],
                    "context": context,
                })

        last_output = output

    # Classification priority (first match wins)
    if has_auth:
        outcome = "api_auth_failure"
    elif has_file and has_correction:
        outcome = "success_after_correction"
    elif has_file and file_turn == num_turns - 1:
        # File in last turn, no corrections
        outcome = "clean_success"
    elif has_file:
        # File delivered but conversation continued
        outcome = "file_then_continued"
    elif not has_file and _is_substantive_answer(last_output):
        outcome = "agent_answered_question"
    elif num_turns > 1 and _ends_with_question(last_output):
        outcome = "stalled_awaiting_user"
    elif num_turns == 1:
        outcome = "single_turn_abandoned"
    else:
        outcome = "other"

    return {
        "outcome": outcome,
        "num_turns": num_turns,
        "has_file_delivery": has_file,
        "file_turn": file_turn,
        "has_auth_failure": has_auth,
        "has_correction": has_correction,
        "correction_details": correction_details,
        "last_output_ends_with_question": _ends_with_question(last_output),
        "last_output_snippet": last_output[:300],
    }


def _classify_stall_question(output: str) -> str:
    """Classify what the agent was asking about when the user stopped."""
    lower = output.lower()
    if any(w in lower for w in ["upload", "attach", "file", "send me", "provide.*file"]):
        return "asking_for_file"
    if any(w in lower for w in ["schema", "template", "format", "column"]):
        return "asking_about_schema"
    return "asking_for_clarification"


def main():
    load_dotenv()
    config = get_project_config()
    host = config.host
    include_internal = os.getenv("INCLUDE_INTERNAL", "").lower() in ("1", "true", "yes")

    now = datetime.now(timezone.utc)
    analysis_start = now - timedelta(days=ANALYSIS_DAYS)
    history_start = now - timedelta(days=HISTORY_DAYS)

    print(f"Project: {config.name}")
    client = get_client(config)

    # Step 1: Fetch full history window
    print(f"Fetching historical traces ({history_start.date()} to {now.date()}) "
          f"to identify first sessions...")
    all_traces = get_traces(client, config, from_timestamp=history_start, to_timestamp=now)
    all_sessions = group_by_session(all_traces)
    print(f"  {len(all_traces)} traces across {len(all_sessions)} sessions\n")

    # Step 2: Find each user's first-ever session
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

        # Failure mode classification
        fm = classify_failure_mode(sess_traces)

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
            "failure_mode": fm["outcome"],
            "has_file_delivery": fm["has_file_delivery"],
            "file_turn": fm["file_turn"],
            "has_auth_failure": fm["has_auth_failure"],
            "has_correction": fm["has_correction"],
            "correction_details": fm["correction_details"],
            "last_output_ends_with_question": fm["last_output_ends_with_question"],
            "last_output_snippet": fm["last_output_snippet"],
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

    df = df_filtered if len(df_filtered) > 0 else df_cat
    label_note = "" if len(df_filtered) > 0 else " (NOTE: includes first sessions)"

    # Build report
    lines: list[str] = []
    def out(s=""):
        lines.append(s)

    out("# Failure Mode Analysis")
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
    out(f"2. Excluded first-ever session per user: {first_excluded} sessions removed")
    out(f"3. **Remaining sessions for analysis: {len(df)}**{label_note}")
    out()
    out("---")
    out()

    total = len(df)

    # Section 1: Outcome distribution
    out("## 1. Outcome Distribution")
    out()
    outcome_counts = df["failure_mode"].value_counts()
    rows = []
    for outcome in OUTCOME_ORDER:
        cnt = int(outcome_counts.get(outcome, 0))
        pct = f"{100 * cnt / total:.1f}%" if total else "0%"
        rows.append([OUTCOME_LABELS[outcome], cnt, pct, OUTCOME_DESCRIPTIONS[outcome]])
    rows.append(["**Total**", f"**{total}**", "**100%**", ""])
    out(markdown_table(["Outcome", "Sessions", "%", "Description"], rows))
    out()

    # Section 2: Outcome by prompt type
    out("## 2. Outcome by Prompt Type")
    out()
    outcome_by_prompt = (
        df.groupby(["prompt_class", "failure_mode"])
        .size()
        .unstack(fill_value=0)
    )
    for col in OUTCOME_ORDER:
        if col not in outcome_by_prompt.columns:
            outcome_by_prompt[col] = 0
    outcome_by_prompt = outcome_by_prompt[OUTCOME_ORDER]
    outcome_by_prompt["Total"] = outcome_by_prompt.sum(axis=1)
    prompt_labels = {"custom": "Custom", "pre_canned": "Pre-canned", "system_auto": "System", "greeting_test": "Greeting"}
    outcome_by_prompt.index = outcome_by_prompt.index.map(lambda x: prompt_labels.get(x, x))
    outcome_by_prompt.index.name = "Prompt Type"
    # Use shorter column names for readability
    short_outcome = {o: OUTCOME_LABELS[o].split(",")[0].split("(")[0].strip() for o in OUTCOME_ORDER}
    outcome_by_prompt.columns = [short_outcome.get(c, c) for c in outcome_by_prompt.columns]
    out(df_to_markdown_table(outcome_by_prompt))
    out()

    # Section 3: Turns-to-value
    out("## 3. Turns to Value (Successful Sessions)")
    out()
    success_modes = ["clean_success", "success_after_correction", "file_then_continued"]
    successful = df[df["failure_mode"].isin(success_modes)]
    if len(successful) > 0:
        successful = successful.copy()
        # file_turn is 0-indexed, display as 1-indexed
        successful["file_turn_display"] = successful["file_turn"].apply(
            lambda x: x + 1 if x is not None and pd.notna(x) else None
        )
        turn_counts = successful["file_turn_display"].value_counts().sort_index()
        rows = []
        for turn_num, cnt in turn_counts.items():
            rows.append([f"Turn {int(turn_num)}", cnt])
        out(markdown_table(["First File Delivery", "Sessions"], rows))
        out()
        valid_turns = successful["file_turn_display"].dropna()
        if len(valid_turns) > 0:
            out(f"- **Average turn of first file delivery**: {valid_turns.mean():.1f}")
            out(f"- **Median**: {int(valid_turns.median())}")
            out()
    else:
        out("No successful file-delivery sessions found.")
        out()

    # Section 4: Correction details
    out("## 4. Correction Details")
    out()
    corrected = df[df["has_correction"]]
    if len(corrected) > 0:
        out(f"**{len(corrected)} sessions** had user corrections.")
        out()
        for _, r in corrected.sort_values("timestamp").iterrows():
            ts = str(r["timestamp"])[:10]
            user = str(r["user_id"] or "?")[:20]
            org = r["org_name"] or "?"
            out(f"### {ts} — {user} ({org})")
            out(f"- **Outcome**: {OUTCOME_LABELS.get(r['failure_mode'], r['failure_mode'])}")
            out(f"- **Trace**: {r['trace_url']}")
            out()
            for cd in r["correction_details"]:
                out(f"**Turn {cd['turn']}** — User correction:")
                out(f"> {cd['user_correction'][:300]}")
                out()
                if cd["agent_output_before"]:
                    out(f"Agent output before correction:")
                    out(f"> {cd['agent_output_before'][:300]}")
                    out()
    else:
        out("No corrections detected.")
        out()

    # Section 5: Stalled sessions
    out("## 5. Stalled Sessions")
    out()
    stalled = df[df["failure_mode"] == "stalled_awaiting_user"]
    if len(stalled) > 0:
        stalled = stalled.copy()
        stalled["stall_type"] = stalled["last_output_snippet"].apply(_classify_stall_question)
        stall_counts = stalled["stall_type"].value_counts()
        stall_labels = {
            "asking_for_file": "Asking for file upload",
            "asking_about_schema": "Asking about schema/format",
            "asking_for_clarification": "Asking for clarification",
        }
        rows = []
        for stype in ["asking_for_file", "asking_about_schema", "asking_for_clarification"]:
            cnt = int(stall_counts.get(stype, 0))
            rows.append([stall_labels[stype], cnt])
        out(markdown_table(["Stall Reason", "Sessions"], rows))
        out()
        out("### What the agent was asking")
        out()
        for _, r in stalled.sort_values("timestamp").iterrows():
            ts = str(r["timestamp"])[:10]
            user = str(r["user_id"] or "?")[:20]
            snippet = r["last_output_snippet"].replace("\n", " ")[:200]
            out(f"- [{ts}] {user}: ...{snippet}")
        out()
    else:
        out("No stalled sessions found.")
        out()

    # Section 6: Auth failures
    out("## 6. Auth Failures")
    out()
    auth_fails = df[df["failure_mode"] == "api_auth_failure"]
    if len(auth_fails) > 0:
        out(f"**{len(auth_fails)} sessions** hit 401/Unauthorized errors.")
        out()
        rows = []
        for _, r in auth_fails.sort_values("timestamp").iterrows():
            ts = str(r["timestamp"])[:10]
            user = str(r["user_id"] or "?")[:20]
            org = r["org_name"] or "?"
            rows.append([ts, user, str(org)[:20], f"[trace]({r['trace_url']})"])
        out(markdown_table(["Date", "User", "Org", "Trace"], rows))
        out()
        out("These are infra issues to flag with the platform team.")
        out()
    else:
        out("No auth failures detected.")
        out()

    # Section 7: Example sessions
    out("## 7. Example Sessions")
    out()
    for outcome in OUTCOME_ORDER:
        subset = df[df["failure_mode"] == outcome].sort_values("timestamp")
        if len(subset) == 0:
            continue
        out(f"### {OUTCOME_LABELS[outcome]}")
        out()
        examples = subset.head(3)
        for _, r in examples.iterrows():
            ts = str(r["timestamp"])[:10]
            user = str(r["user_id"] or "?")[:20]
            org = r["org_name"] or "?"
            inp = str(r["first_input"]).replace("\n", " ")[:200]
            out(f"- **{ts}** — {user} ({org})")
            out(f"  - Turns: {r['num_turns']} | Prompt: {r['prompt_class']}")
            out(f"  - Input: {inp}")
            out(f"  - [Trace]({r['trace_url']})")
            out()

    # Appendix
    out("---")
    out()
    out("## Appendix: Filter Summary")
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

    # Build data dict
    # Serialize correction_details to avoid issues with nested objects
    session_records = df[[
        "session_id", "user_id", "user_name", "category",
        "org_id", "org_name", "environment",
        "prompt_class", "suggestion_name",
        "failure_mode", "has_file_delivery", "file_turn",
        "has_auth_failure", "has_correction",
        "correction_details",
        "num_turns", "first_input", "last_output_snippet",
        "trace_url", "trace_id", "is_first_session",
    ]].to_dict(orient="records")

    data = {
        "generated": now.isoformat(),
        "analysis_window": {"from": analysis_start.isoformat(), "to": now.isoformat()},
        "history_window": {"from": history_start.isoformat(), "to": now.isoformat()},
        "include_internal": include_internal,
        "filters": {
            "total_sessions": len(df_all),
            "excluded_non_customer": len(df_all) - len(df_cat),
            "excluded_first_sessions": len(df_cat) - len(df_filtered),
            "remaining": len(df),
        },
        "outcome_distribution": {
            outcome: int(outcome_counts.get(outcome, 0))
            for outcome in OUTCOME_ORDER
        },
        "sessions": session_records,
    }

    save_results("failure_modes", report_md, data, project=config.name)


if __name__ == "__main__":
    main()
