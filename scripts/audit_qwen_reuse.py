"""Audit whether pre-freeze Qwen checkpoints can be reused without API calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


EXPECTED_CASES = (
    "real_dropout", "real_fault", "virtual_fault", "virtual_dropout",
    "label_dropout", "hetero_abrupt_dropout", "fault_dropout",
    "real_hetero", "virtual_hetero",
)
EXPECTED_SEEDS = (0, 44, 56)
ALLOWED_MANIFEST_DIFFS = {
    "frozen_on", "protocol.response_window_length", "integrity.manifest_sha256",
}


def nested_diffs(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(nested_diffs(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def roots(values):
    parsed = []
    for value in values:
        mode, separator, raw_path = value.partition("=")
        if not separator or mode not in {"none", "low", "xhigh"}:
            raise ValueError(f"invalid --run-root {value!r}; expected MODE=PATH")
        parsed.append((mode, Path(raw_path).resolve()))
    return parsed


def load_old_manifest(benchmark_root: Path, revision: str):
    raw = subprocess.check_output(
        ["git", "-c", f"safe.directory={benchmark_root.as_posix()}",
         "show", f"{revision}:src/b3_suite_manifest.json"],
        cwd=benchmark_root,
    )
    return json.loads(raw)


def checkpoint_for(root: Path, episode_id: str, mode: str):
    directory = root / "private" / "checkpoints" / episode_id
    preferred = directory / f"qwen3p8_flash_{mode}.json"
    if preferred.exists():
        return preferred
    candidates = [
        path for path in directory.glob("qwen3p8_flash*.json")
        if not path.name.endswith(".run_integrity.json")
    ]
    return candidates[0] if len(candidates) == 1 else None


def failure_type(root: Path, episode_id: str, mode: str):
    directory = root / "private" / "checkpoints" / episode_id
    paths = list(directory.glob(f"qwen3p8_flash*{mode}*.run_integrity.json"))
    if not paths:
        paths = list(directory.glob("*.run_integrity.json"))
    if not paths:
        return "MISSING_CHECKPOINT"
    try:
        return load_json(paths[0]).get("integrity", {}).get(
            "failure_type", "INVALID_CHECKPOINT"
        )
    except (OSError, json.JSONDecodeError):
        return "INVALID_CHECKPOINT"


def audit_checkpoint(path: Path, plan: dict, mode: str, context: dict):
    from agent_decision_audit import audit_episode
    from b3_fingerprint import canonical_sha256
    from b3_episode_plan import plan_sha256
    from observation_contract import (
        PRIVATE_SCORER_SCHEMA, public_observation, serialize_llm_observation,
    )

    reasons = []
    checkpoint = load_json(path)
    fingerprint = checkpoint.get("run_fingerprint", {})
    spec = checkpoint.get("agent_spec", {})
    history = checkpoint.get("history", {})
    telemetry = history.get("telemetry", [])
    acc = history.get("acc", [])
    num_clients = int(fingerprint.get("config", {}).get("num_clients", 20))

    def require(condition, reason):
        if not condition:
            reasons.append(reason)

    require(checkpoint.get("episode_id") == plan.get("episode_id"), "episode_mismatch")
    require(plan.get("plan_sha256") == plan_sha256(plan), "plan_sha_invalid")
    require(fingerprint.get("plan_sha256") == plan.get("plan_sha256"), "plan_sha_mismatch")
    require(fingerprint.get("partition_sha256") == plan.get("partition_sha256"),
            "partition_sha_mismatch")
    require(fingerprint.get("rounds") == 80, "not_80_rounds")
    require(len(telemetry) == 80 and len(acc) == 80, "incomplete_history")
    require([row.get("round") for row in telemetry] == list(range(80)),
            "round_sequence_invalid")
    require(fingerprint.get("llm_prompt_sha256") == context["prompt_sha"],
            "prompt_sha_mismatch")
    require(spec.get("prompt_sha256") == context["prompt_sha"],
            "agent_prompt_sha_mismatch")
    require(spec.get("model") == "qwen3.8-flash", "model_mismatch")
    require(spec.get("reasoning_effort") == mode, "reasoning_effort_mismatch")
    require(spec.get("decision_cadence") == 1, "decision_cadence_mismatch")
    require(checkpoint.get("run_fingerprint_sha256") == fingerprint.get("sha256"),
            "fingerprint_reference_mismatch")
    require(
        fingerprint.get("sha256") == canonical_sha256({
            key: value for key, value in fingerprint.items() if key != "sha256"
        }),
        "fingerprint_sha_invalid",
    )
    manifest_sha = fingerprint.get("manifest_sha256")
    require(manifest_sha in context["compatible_manifest_shas"],
            "trajectory_manifest_incompatible")

    observation_errors = Counter()
    private_keys = set(PRIVATE_SCORER_SCHEMA)
    for row in telemetry:
        try:
            full = json.loads(row["full_public_observation_json"])
            compact = json.loads(row["public_observation_json"])
            public_observation(full)
            memory = {
                key: compact[key] for key in (
                    "last_decision", "active_action_ages",
                    "telemetry_delta_since_last_decision",
                )
            }
            if serialize_llm_observation(full, memory) != row["public_observation_json"]:
                observation_errors["llm_observation_contract_mismatch"] += 1
            if private_keys & set(compact):
                observation_errors["private_field_in_llm_observation"] += 1
            if "client_utilities" in compact or "feddrift_trigger" in compact:
                observation_errors["removed_field_in_llm_observation"] += 1
            if row.get("llm_request_sha256") != hashlib.sha256(
                    row["public_observation_json"].encode("utf-8")).hexdigest():
                observation_errors["request_sha_mismatch"] += 1
            expected = (
                round(row["available_clients_count"] / num_clients, 3),
                round(row["planned_clients_count"] / num_clients, 3),
                round(max(0, row["planned_clients_count"] -
                          row["participating_clients_count"]) / num_clients, 3),
            )
            observed = (
                full["availability_rate"], full["planned_participation_rate"],
                full["participation_gap"],
            )
            if expected != observed:
                observation_errors["availability_contract_mismatch"] += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            observation_errors["observation_invalid"] += 1
    reasons.extend(sorted(observation_errors))

    hidden = history.get("hidden_event_log", [])
    require(len(hidden) == 1, "hidden_event_log_invalid")
    if len(hidden) == 1:
        event = hidden[0]
        expected = plan.get("hidden_event", {})
        for key in (
            "active_causes", "affected_clients", "cause_intervals", "event_schedule",
        ):
            require(event.get(key) == expected.get(key), f"hidden_{key}_mismatch")
        require(event.get("plan_sha256") == plan.get("plan_sha256"),
                "hidden_plan_sha_mismatch")
        require(event.get("partition_sha256") == plan.get("partition_sha256"),
                "hidden_partition_sha_mismatch")

    errors = sum(bool(row.get(key)) for row in telemetry for key in (
        "agent_interface_error", "llm_api_error", "llm_parse_error", "llm_budget_error",
    ))
    require(errors == 0, "agent_interface_error")

    audit = audit_episode(
        "qwen-pre-freeze-reuse-audit", checkpoint.get("agent", path.stem),
        spec, plan, history, context["manifest"],
    )
    require(audit["integrity"]["run_status"] == "VALID", "formal_audit_invalid")
    event = next((row for row in audit["events"]
                  if row["transition_type"] != "INIT"), None)
    require(event is not None, "scored_event_missing")
    window = [] if event is None else [
        row for row in audit["decisions"]
        if row["event_id"] == event["event_id"] and
        int(event["first_due_round"]) <= int(row["decision_round"]) <=
        int(event["emit_deadline"])
    ]
    correct = [
        row for row in window
        if row["decision_label"] in {"CORRECT", "NO_OP_CORRECT"}
    ]
    extra = [row for row in window if row["unnecessary_actions"]]

    return {
        "trajectory_reusable": not reasons,
        "reasons": sorted(set(reasons)),
        "source_sha256": fingerprint.get("source_sha256", ""),
        "manifest_sha256": manifest_sha or "",
        "prompt_sha256": fingerprint.get("llm_prompt_sha256", ""),
        "temperature": spec.get("temperature"),
        "formal_temp0_compatible": not reasons and spec.get("temperature") == 0.0,
        "success_at_10": bool(correct) and not extra,
        "first_correct_round": correct[0]["decision_round"] if correct else "",
        "ended_correct": bool(window and window[-1]["decision_label"] in
                              {"CORRECT", "NO_OP_CORRECT"}),
        "correct_decisions_at_10": len(correct),
        "extra_decisions_at_10": len(extra),
        "api_calls": sum(bool(row.get("llm_called")) for row in telemetry),
        "prompt_tokens": sum(int(row.get("llm_prompt_tokens", 0)) for row in telemetry),
        "completion_tokens": sum(
            int(row.get("llm_completion_tokens", 0)) for row in telemetry
        ),
    }


def audit_candidate(root: Path, plan_path: Path, mode: str, context: dict):
    plan = load_json(plan_path)
    case_id, seed = plan.get("case_id"), plan.get("training_seed")
    base = {
        "mode": mode, "case": case_id, "seed": seed, "run_root": str(root),
        "checkpoint": "", "selected": False,
    }
    path = checkpoint_for(root, plan["episode_id"], mode)
    if path is None:
        reason = failure_type(root, plan["episode_id"], mode)
        return {**base, "trajectory_reusable": False, "reasons": [reason],
                "source_sha256": "", "manifest_sha256": "", "prompt_sha256": "",
                "temperature": "", "formal_temp0_compatible": False,
                "success_at_10": False, "first_correct_round": "",
                "ended_correct": "", "correct_decisions_at_10": 0,
                "extra_decisions_at_10": 0, "api_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "mtime": plan_path.stat().st_mtime}
    try:
        details = audit_checkpoint(path, plan, mode, context)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        details = {
            "trajectory_reusable": False,
            "reasons": [f"audit_exception:{type(exc).__name__}"],
            "source_sha256": "", "manifest_sha256": "", "prompt_sha256": "",
            "temperature": "", "formal_temp0_compatible": False,
            "success_at_10": False, "first_correct_round": "",
            "ended_correct": "", "correct_decisions_at_10": 0,
            "extra_decisions_at_10": 0, "api_calls": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
        }
    return {**base, **details, "checkpoint": str(path), "mtime": path.stat().st_mtime}


def choose(rows):
    grouped = {}
    for row in rows:
        key = row["mode"], row["case"], row["seed"]
        score = (bool(row["trajectory_reusable"]), row["api_calls"], row["mtime"])
        if key not in grouped or score > grouped[key][0]:
            grouped[key] = score, row
    selected = []
    for _, row in grouped.values():
        row["selected"] = True
        selected.append(row)
    return sorted(selected, key=lambda row: (row["mode"], row["seed"], row["case"]))


def write_outputs(output: Path, rows, selected, context):
    output.mkdir(parents=True, exist_ok=True)
    public_rows = [{key: value for key, value in row.items() if key != "mtime"}
                   for row in rows]
    selected_public = [{key: value for key, value in row.items() if key != "mtime"}
                       for row in selected]
    summary = {}
    for mode in ("none", "low", "xhigh"):
        mode_rows = [row for row in selected if row["mode"] == mode]
        summary[mode] = {
            "covered": len(mode_rows),
            "reusable": sum(bool(row["trajectory_reusable"]) for row in mode_rows),
            "rerun_required": sum(not row["trajectory_reusable"] for row in mode_rows),
            "success_at_10": sum(bool(row["success_at_10"])
                                  for row in mode_rows if row["trajectory_reusable"]),
            "formal_temp0_compatible": sum(bool(row["formal_temp0_compatible"])
                                            for row in mode_rows),
        }
    payload = {
        "audit_version": "qwen-pre-freeze-reuse-v1",
        "frozen_commit": context["freeze_commit"],
        "frozen_manifest_sha256": context["manifest_sha"],
        "prompt_sha256": context["prompt_sha"],
        "manifest_diff_paths": context["manifest_diff_paths"],
        "summary": summary, "selected_results": selected_public,
        "all_candidates": public_rows,
    }
    (output / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "mode", "case", "seed", "trajectory_reusable", "formal_temp0_compatible",
        "success_at_10", "first_correct_round", "ended_correct",
        "correct_decisions_at_10", "extra_decisions_at_10", "api_calls",
        "prompt_tokens", "completion_tokens", "temperature", "reasons",
        "source_sha256", "manifest_sha256", "prompt_sha256", "checkpoint",
    ]
    with (output / "audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in selected_public:
            writer.writerow({
                key: "|".join(row[key]) if key == "reasons" else row.get(key, "")
                for key in fields
            })

    lines = [
        "# Qwen 旧结果复用审计", "",
        f"- 冻结 commit：`{context['freeze_commit']}`",
        f"- 冻结 manifest：`{context['manifest_sha']}`",
        f"- Prompt SHA：`{context['prompt_sha']}`",
        f"- 旧/新 manifest 实质差异：`{', '.join(context['manifest_diff_paths'])}`",
        "- `trajectory_reusable` 表示无需 API 即可在冻结10轮口径下复用。",
        "- `formal_temp0_compatible` 额外要求旧调用本身使用 temperature=0.0。",
        "", "| Mode | Covered | Reusable | Rerun | Success@10 | Temp=0 formal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, values in summary.items():
        lines.append(
            f"| {mode} | {values['covered']} | {values['reusable']} | "
            f"{values['rerun_required']} | {values['success_at_10']} | "
            f"{values['formal_temp0_compatible']} |"
        )
    lines.extend(["", "## 逐条结果", "",
                  "| Mode | Case | Seed | Reuse | Success@10 | First | Ended | Reasons |",
                  "|---|---|---:|---|---|---:|---|---|"])
    for row in selected_public:
        lines.append(
            f"| {row['mode']} | {row['case']} | {row['seed']} | "
            f"{'YES' if row['trajectory_reusable'] else 'NO'} | "
            f"{row['success_at_10']} | {row['first_correct_round']} | "
            f"{row['ended_correct']} | {'; '.join(row['reasons'])} |"
        )
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--run-root", action="append", default=[], metavar="MODE=PATH")
    parser.add_argument("--merge-input", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-manifest-revision", default="16ca7ad")
    args = parser.parse_args()
    if args.merge_input:
        audits = [load_json(path) for path in args.merge_input]
        candidates = [
            {**row, "mtime": 0.0}
            for audit in audits for row in audit["selected_results"]
        ]
        selected = choose(candidates)
        first = audits[0]
        context = {
            "freeze_commit": first["frozen_commit"],
            "manifest_sha": first["frozen_manifest_sha256"],
            "prompt_sha": first["prompt_sha256"],
            "manifest_diff_paths": first["manifest_diff_paths"],
        }
        write_outputs(args.output.resolve(), candidates, selected, context)
        print("QWEN_REUSE_AUDIT_MERGE_OK", len(selected))
        return
    if args.benchmark_root is None or not args.run_root:
        parser.error("--benchmark-root and --run-root are required unless merging")
    benchmark_root = args.benchmark_root.resolve()
    sys.path.insert(0, str(benchmark_root / "src"))

    from b3_fingerprint import source_sha256
    from b3_manifest import load_manifest, manifest_sha256
    from llm_backend import B3_SYSTEM_PROMPT

    manifest = load_manifest(benchmark_root / "src" / "b3_suite_manifest.json")
    old_manifest = load_old_manifest(benchmark_root, args.old_manifest_revision)
    manifest_diff_paths = nested_diffs(old_manifest, manifest)
    if set(manifest_diff_paths) - ALLOWED_MANIFEST_DIFFS:
        raise RuntimeError(f"manifest has trajectory-changing diffs: {manifest_diff_paths}")
    old_sha = manifest_sha256(old_manifest)
    current_sha = manifest_sha256(manifest)
    context = {
        "manifest": manifest,
        "manifest_sha": current_sha,
        "compatible_manifest_shas": {old_sha, current_sha},
        "manifest_diff_paths": manifest_diff_paths,
        "prompt_sha": hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "freeze_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={benchmark_root.as_posix()}",
             "rev-parse", "HEAD"], cwd=benchmark_root, text=True,
        ).strip(),
        "source_sha": source_sha256(benchmark_root),
    }
    candidates = []
    for mode, root in roots(args.run_root):
        plan_dir = root / "private" / "private_plans"
        for plan_path in sorted(plan_dir.glob("*.json")):
            plan = load_json(plan_path)
            if plan.get("case_id") in EXPECTED_CASES and plan.get("training_seed") in EXPECTED_SEEDS:
                candidates.append(audit_candidate(root, plan_path, mode, context))
    selected = choose(candidates)
    write_outputs(args.output.resolve(), candidates, selected, context)
    print("QWEN_REUSE_AUDIT_OK", json.dumps({
        mode: {
            "covered": sum(row["mode"] == mode for row in selected),
            "reusable": sum(row["mode"] == mode and row["trajectory_reusable"]
                            for row in selected),
        }
        for mode in ("none", "low", "xhigh")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
