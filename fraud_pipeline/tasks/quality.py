"""Data Quality snapshot, nine rules and an always-written HTML report."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fraud_pipeline.features import REQUIRED
from fraud_pipeline.storage import atomic_bytes, connect, iter_frames, root, run_dir, write_json


def snapshot_quality(run_id):
    """Read a consistent mart snapshot in chunks; save aggregate evidence.

    One read transaction isolates concurrent supply. Only summary statistics
    are retained, so a 786431-row CSV does not create a full copy each run.
    Repeats reuse the summary and never mutate data or model state.
    """
    path = run_dir(run_id, "dq") / "snapshot.json"
    if not path.exists():
        with connect() as c:
            c.execute("BEGIN")
            summary = empty_summary()
            for frame in iter_frames(c):
                add_frame(summary,frame)
            count,unique = c.execute("SELECT count(*),count(DISTINCT transaction_id) FROM labeled_transactions").fetchone()
            summary["duplicates"] = count-unique
        write_json(path, summary)
    return {"path": str(path), "directory": str(path.parent), "run_id": run_id}


def empty_summary():
    return dict(rows=0,missing=[],nulls=0,duplicates=0,bad_times=0,bad_amounts=0,
                bad_coordinates=0,bad_population=0,bad_labels=0,months=[],classes={})


def add_frame(summary,frame):
    """Accumulate exact counts; retain at most two class values per month."""
    required = set(REQUIRED+["transaction_id","target"])
    summary["rows"] += len(frame)
    summary["missing"] = sorted(set(summary["missing"]) | (required-set(frame.columns)))
    summary["nulls"] += int(frame.reindex(columns=sorted(required)).isna().sum().sum())
    if required-set(frame.columns):
        return
    times = pd.to_datetime(frame.transaction_time,errors="coerce",utc=True,format="ISO8601")
    summary["bad_times"] += int(times.isna().sum())
    amount = pd.to_numeric(frame.amount,errors="coerce")
    summary["bad_amounts"] += int((~np.isfinite(amount)|(amount<=0)).sum())
    for col,limit in [("lat",90),("lon",180),("merchant_lat",90),("merchant_lon",180)]:
        value = pd.to_numeric(frame[col],errors="coerce")
        summary["bad_coordinates"] += int((~np.isfinite(value)|(value.abs()>limit)).sum())
    population = pd.to_numeric(frame.population_city,errors="coerce")
    summary["bad_population"] += int((~np.isfinite(population)|(population<0)).sum())
    summary["bad_labels"] += int((~frame.target.isin([0,1])).sum())
    month = times.dt.strftime("%Y-%m")
    summary["months"] = sorted(set(summary["months"])|set(month.dropna()))
    for name,group in frame.assign(month=month).groupby("month"):
        present = {int(v) for v in group.target if v in (0,1)}
        summary["classes"][name] = sorted(set(summary["classes"].get(name,[]))|present)


def rules_from_summary(summary):
    rules = []
    def add(name, passed, detail):
        rules.append(dict(rule=name, passed=bool(passed), detail=str(detail)))
    missing = summary["missing"]
    add("Required schema and nonempty table", not missing and summary["rows"]>0,
        f"rows={summary['rows']}, missing={missing}")
    if missing or not summary["rows"]:
        return rules
    nulls = summary["nulls"]
    add("Required values are not null", nulls == 0, f"null cells={nulls}")
    duplicates = summary["duplicates"]
    add("Transaction IDs are unique", duplicates == 0, f"duplicates={duplicates}")
    bad_times = summary["bad_times"]
    add("Transaction times parse", bad_times == 0, f"invalid timestamps={bad_times}")
    invalid = summary["bad_amounts"]
    add("Amount is finite and positive", invalid == 0, f"invalid amounts={invalid}")
    invalid_coords = summary["bad_coordinates"]
    add("Geographic coordinates in range", invalid_coords == 0, f"invalid coordinates={invalid_coords}")
    bad_population = summary["bad_population"]
    add("Population is finite and nonnegative", bad_population == 0, f"invalid populations={bad_population}")
    bad_labels = summary["bad_labels"]
    bad_months = [m for m in summary["months"] if summary["classes"].get(m)!=[0,1]]
    add("Binary labels and both classes each month", bad_labels == 0 and not bad_months,
        f"invalid labels={bad_labels}, single-class months={bad_months}")
    months = summary["months"]
    expected = [str(m) for m in pd.period_range(months[0], months[-1], freq="M")] if months else []
    add("Consecutive calendar months", bool(months) and months == expected, f"months={months}")
    return rules


def check_rules(frame):
    summary = empty_summary()
    add_frame(summary,frame)
    summary["duplicates"] = int(frame.transaction_id.duplicated().sum()) if "transaction_id" in frame else 0
    return rules_from_summary(summary)


def evaluate_quality(info):
    """Read the DQ snapshot; write JSON rule results, including failed rules.

    Pure evaluation is repeatable and does not fail before the HTML task runs.
    """
    summary = json.loads(Path(info["path"]).read_text())
    results = dict(run_id=info["run_id"], rows=summary["rows"], rules=rules_from_summary(summary))
    write_json(Path(info["directory"])/"results.json", results)
    return info


def render_quality(info):
    """Read DQ JSON; atomically publish versioned HTML plus latest.html.

    Repeats replace the same report. Failure results are published before the
    enforcement task can mark the DAG failed, preserving evidence for the UI.
    """
    result = json.loads((Path(info["directory"])/"results.json").read_text())
    escaped = lambda s: html.escape(str(s))
    rows = "".join(f"<tr><td>{escaped(r['rule'])}</td><td class=\"{'ok' if r['passed'] else 'bad'}\">"
                   f"{'PASS' if r['passed'] else 'FAIL'}</td><td>{escaped(r['detail'])}</td></tr>"
                   for r in result["rules"])
    content = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Fraud Data Quality</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#1f2937}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #d9d9d9;padding:12px;text-align:left}"
        "th{background:#eaf0f7}.ok{color:#137333}.bad{color:#b91c1c}td{overflow-wrap:anywhere}</style>"
        f"<h1>Fraud Data Quality</h1><p>Run: {escaped(result['run_id'])}</p>"
        f"<p>Snapshot rows: {result['rows']}</p><table><tr><th>Rule</th><th>Result</th><th>Evidence</th></tr>"
        f"{rows}</table></html>")
    reports = root()/"reports"
    name = Path(info["directory"]).name + ".html"
    for path in [reports/name, reports/"latest.html"]:
        atomic_bytes(path, lambda f: f.write(content.encode()))
    return dict(info, report=str(reports/name))


def enforce_quality(info):
    """Read saved results and fail if any rule failed; reports remain available.

    Repeats only re-check immutable results and cannot modify the dataset.
    """
    result = json.loads((Path(info["directory"])/"results.json").read_text())
    failed = [r["rule"] for r in result["rules"] if not r["passed"]]
    if failed:
        raise ValueError(f"Data quality failed: {failed}; report: {info['report']}")
    return info["report"]
