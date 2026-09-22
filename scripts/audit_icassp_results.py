"""Audit saved ICASSP checkpoints with the frozen event scorer; no training/API."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_decision_audit import audit_episode  # noqa: E402
from b3_episode_plan import plan_sha256  # noqa: E402
from b3_fingerprint import canonical_sha256  # noqa: E402
from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256  # noqa: E402


PROMPT_SHA = "01779b937baf079ebc7b9c37a757e70bc0f40f83e784170b15850fa54f5ae291"
BASE_MANIFEST_SHA = "af5afd16150f4378edeace0ad00151904c7fb8c6d15e8bcebb9ee9a8e3eb540d"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_for(scan_root: Path, base_manifest_path: Path):
    atomic = scan_root / "private" / "atomic_manifest.json"
    return load_manifest(atomic if atomic.exists() else base_manifest_path)


def journal_metrics(checkpoint: Path):
    path = checkpoint.with_suffix(".llm_journal.jsonl")
    if not path.exists():
        return 0, 0, 0, []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return (
        sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in rows),
        sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in rows),
        len(rows),
        [float(row.get("latency_ms", 0.0)) for row in rows],
    )


def audit_root(scan_root: Path, label: str, base_manifest_path: Path):
    manifest = manifest_for(scan_root, base_manifest_path)
    manifest_sha = manifest_sha256(manifest)
    atomic = manifest.get("atomic_extension")
    if atomic:
        assert atomic["base_manifest_sha256"] == BASE_MANIFEST_SHA
    else:
        assert manifest_sha == BASE_MANIFEST_SHA

    plans = list(scan_root.rglob("private_plans/*.json"))
    rows, events = [], []
    for plan_path in plans:
        plan = read_json(plan_path)
        assert plan["plan_sha256"] == plan_sha256(plan), plan_path
        checkpoint_dir = plan_path.parent.parent / "checkpoints" / plan["episode_id"]
        for path in sorted(checkpoint_dir.glob("*.json")):
            if path.name.endswith(".run_integrity.json"):
                payload = read_json(path)
                integrity = payload["integrity"]
                rows.append({
                    "suite": label, "case": plan["case_id"],
                    "seed": int(plan["training_seed"]),
                    "agent": path.name.removesuffix(".run_integrity.json"),
                    "model": "deterministic", "effort": "", "temperature": "",
                    "status": "TERMINAL", "failure": integrity["failure_type"],
                    "rerun_required": integrity["rerun_required"], "rounds": 0,
                    "success5": "", "success10": "", "ended": "",
                    "correct_decisions": 0, "decisions": 0, "over": 0,
                    "invalid": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "api_calls": 0, "mean_latency_ms": "", "p95_latency_ms": "",
                    "source_sha": "", "manifest_sha": manifest_sha,
                    "prompt_sha": PROMPT_SHA,
                })
                continue

            payload = read_json(path)
            fingerprint = payload["run_fingerprint"]
            assert payload["episode_id"] == plan["episode_id"], path
            assert fingerprint["plan_sha256"] == plan["plan_sha256"], path
            assert fingerprint["manifest_sha256"] == manifest_sha, path
            assert fingerprint["sha256"] == canonical_sha256({
                key: value for key, value in fingerprint.items() if key != "sha256"
            }), path
            assert payload["run_fingerprint_sha256"] == fingerprint["sha256"], path
            assert fingerprint["rounds"] == 80 and len(payload["history"]["telemetry"]) == 80, path
            spec = payload["agent_spec"]
            model = spec.get("model", "deterministic")
            if model != "deterministic":
                assert spec.get("prompt_sha256") == PROMPT_SHA, path
            prompt, completion, calls, latencies = journal_metrics(path)
            latency_sorted = sorted(latencies)
            p95 = (latency_sorted[min(len(latency_sorted) - 1,
                                       int(0.95 * len(latency_sorted)))]
                   if latency_sorted else "")
            try:
                result = audit_episode(
                    "icassp-results-audit", payload["agent"], spec, plan,
                    payload["history"], manifest,
                )
                integrity = result["integrity"]
                activation = [event for event in result["events"]
                              if event["transition_type"] != "INIT"]
                # Startup heterogeneity is represented by the INIT event.
                if not activation and len(result["events"]) == 1:
                    activation = result["events"]
                event = activation[0] if len(activation) == 1 else None
            except ValueError:
                result = {"events": []}
                integrity = {
                    "run_status": "INVALID", "failure_type": "AUDIT_PLAN_INCONSISTENT",
                    "rerun_required": True,
                }
                event = None
            valid = integrity["run_status"] == "VALID" and event is not None
            delay = int(event["decision_delay"]) if valid and event["decision_delay"] != "" else None
            rows.append({
                "suite": label, "case": plan["case_id"],
                "seed": int(plan["training_seed"]), "agent": payload["agent"],
                "model": model, "effort": spec.get("reasoning_effort", ""),
                "temperature": spec.get("temperature", ""),
                "status": integrity["run_status"], "failure": integrity["failure_type"],
                "rerun_required": integrity["rerun_required"], "rounds": 80,
                "success5": valid and delay is not None and delay <= 4,
                "success10": valid and event["timing_status"] == "ON_TIME",
                "ended": valid and bool(event["ended_correct"]),
                "correct_decisions": int(event["correct_decision_count"]) if valid else 0,
                "decisions": int(event["decision_count"]) if valid else 0,
                "over": int(event["over_intervention_decision_count"]) if valid else 0,
                "invalid": int(event["invalid_bundle_count"]) if valid else 0,
                "prompt_tokens": prompt, "completion_tokens": completion,
                "api_calls": calls,
                "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else "",
                "p95_latency_ms": round(float(p95), 3) if latencies else "",
                "source_sha": fingerprint["source_sha256"],
                "manifest_sha": manifest_sha,
                "prompt_sha": fingerprint["llm_prompt_sha256"],
            })
            if valid:
                events.append({"suite": label, "agent": payload["agent"], **event})
    return rows, events, len(plans)


def aggregate(rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["agent"], row["case"])].append(row)
    output = []
    for (suite, agent, case), group in sorted(grouped.items()):
        valid = [row for row in group if row["status"] == "VALID"]
        decisions = sum(int(row["decisions"]) for row in valid)
        output.append({
            "suite": suite, "agent": agent, "case": case,
            "planned": len(group), "valid": len(valid),
            "success5": sum(row["success5"] is True for row in valid),
            "success10": sum(row["success10"] is True for row in valid),
            "ended": sum(row["ended"] is True for row in valid),
            "correct_decision_rate": round(
                sum(int(row["correct_decisions"]) for row in valid) / decisions, 6
            ) if decisions else "",
            "over_intervention_rate": round(
                sum(int(row["over"]) for row in valid) / decisions, 6
            ) if decisions else "",
            "invalid_bundle_rate": round(
                sum(int(row["invalid"]) for row in valid) / decisions, 6
            ) if decisions else "",
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in valid),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in valid),
            "api_calls": sum(int(row["api_calls"]) for row in valid),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", type=Path)
    parser.add_argument("--atomic-root", type=Path)
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    all_rows, all_events, plan_counts = [], [], {}
    if args.deterministic:
        rows, events, count = audit_root(
            args.deterministic, "deterministic360", args.manifest
        )
        all_rows += rows; all_events += events; plan_counts["deterministic360"] = count
    if args.atomic_root:
        for suite in sorted(args.atomic_root.glob("atomic_*_v2")):
            if (suite / "private" / "atomic_manifest.json").exists():
                rows, events, count = audit_root(suite, suite.name, args.manifest)
                all_rows += rows; all_events += events; plan_counts[suite.name] = count
    for suite in args.extra_root:
        rows, events, count = audit_root(suite, suite.name, args.manifest)
        all_rows += rows; all_events += events; plan_counts[suite.name] = count
    summary = aggregate(all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "runs.csv", all_rows)
    write_csv(args.out / "events.csv", all_events)
    write_csv(args.out / "per_case.csv", summary)
    (args.out / "audit.json").write_text(json.dumps({
        "plan_counts": plan_counts,
        "runs": len(all_rows),
        "valid": sum(row["status"] == "VALID" for row in all_rows),
        "terminal": sum(row["status"] == "TERMINAL" for row in all_rows),
        "invalid": sum(row["status"] not in {"VALID", "TERMINAL"} for row in all_rows),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "plans": plan_counts, "runs": len(all_rows),
        "valid": sum(row["status"] == "VALID" for row in all_rows),
        "terminal": sum(row["status"] == "TERMINAL" for row in all_rows),
        "invalid": sum(row["status"] not in {"VALID", "TERMINAL"} for row in all_rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
