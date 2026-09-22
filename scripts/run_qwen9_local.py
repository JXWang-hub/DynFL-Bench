"""Run selected Qwen B3 cases with resumable checkpoints."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents import B3_VISIBLE_ACTIONS  # noqa: E402
from b3_composite import _configured_llm_agent, _write_json, prepare_episode  # noqa: E402
from b3_fingerprint import run_fingerprint, source_sha256  # noqa: E402
from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256  # noqa: E402
from llm_backend import B3_SYSTEM_PROMPT  # noqa: E402
from scripts.run_atomic_llm import FROZEN_MANIFEST_SHA  # noqa: E402
from scripts.run_b3_formal_eval import (  # noqa: E402
    LOW_REASONING_MODELS, _run_or_record_failure, _terminal_failure,
)


CASES = (
    "real_dropout", "real_fault", "virtual_fault", "virtual_dropout",
    "label_dropout", "hetero_abrupt_dropout", "fault_dropout",
    "real_hetero", "virtual_hetero", "staggered",
)
SEEDS = (0, 44, 56)
DEFAULT_OUT = ROOT / "results"
XHIGH_TIMEOUT_SECONDS = 600.0


def action_set(row):
    return set(row.get("action_bundle", "no_op").split("|")) - {"no_op"}


def journal_calls(checkpoint):
    path = checkpoint.with_suffix(".llm_journal.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_effort(args, reasoning_effort, jobs):
    manifest = load_manifest(DEFAULT_MANIFEST)
    base_sha = manifest_sha256(manifest)
    if base_sha != FROZEN_MANIFEST_SHA:
        raise RuntimeError(
            f"formal Qwen run requires frozen manifest {FROZEN_MANIFEST_SHA}; got {base_sha}"
        )
    out = args.out_dir / f"qwen9_candidate_{reasoning_effort}_remaining_v1"
    model = {**next(
        item for item in LOW_REASONING_MODELS["models"]
        if item["model"] == "qwen3.8-flash"
    ), "name": f"qwen3p8_flash_{reasoning_effort}",
        "reasoning_effort": reasoning_effort, "temperature": 0.6}
    if not os.environ.get(model["api_key_env"]):
        raise RuntimeError(f"missing {model['api_key_env']}")

    response_window_length = int(manifest["protocol"]["response_window_length"])
    prompt_sha = hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    implementation_sha = source_sha256()
    results = []
    total = len(jobs)
    for seed, case_id in jobs:
        plan_path = out / "private" / "private_plans" / f"{case_id}_s{seed}.json"
        cfg, data, plan = prepare_episode(
            case_id, seed, False, False, plan_path=plan_path,
            resume=plan_path.exists(),
        )
        policy = {
            **manifest["protocol"]["phase6"]["agent_policy"],
            "decision_cadence": int(cfg.decide_every),
            "max_calls": int(cfg.rounds),
        }
        if reasoning_effort == "xhigh":
            policy["timeout_seconds"] = XHIGH_TIMEOUT_SECONDS
        agent, spec = _configured_llm_agent(
            model, policy, implementation_sha, ablation_group=args.ablation_group
        )
        checkpoint = (
            out / "private" / "checkpoints" / plan["episode_id"] /
            f"{agent.name}.json"
        )
        fingerprint = run_fingerprint(
            manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
            data=data, agent=agent, synthetic=False, smoke=False,
            prompt_sha=prompt_sha,
        )
        history = _run_or_record_failure(
            checkpoint, agent, cfg, data, plan, fingerprint, spec,
            "QWEN9_LOCAL", f"qwen9-candidate-{reasoning_effort}-v1",
        )
        expected = set(plan["hidden_event"]["canonical_bundle"])
        if history is None:
            terminal = _terminal_failure(checkpoint, fingerprint["sha256"])["integrity"]
            calls = journal_calls(checkpoint)
            result = {
                "case": case_id, "seed": seed, "expected": sorted(expected),
                "first_exact_round": None, "success_at_10": False,
                "exact_decisions_at_10": 0, "extra_action_rounds_at_10": [],
                "correct_rounds_50_79": 0, "round_79_actions": [],
                "api_calls": len(calls),
                "prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in calls),
                "completion_tokens": sum(
                    int(row["usage"]["completion_tokens"]) for row in calls
                ),
                "mean_latency_ms": round(
                    sum(float(row["latency_ms"]) for row in calls) / max(1, len(calls)), 3
                ),
                "interface_error_rounds": [],
                "terminal_failure": {
                    "type": terminal["failure_type"],
                    "round": terminal["failure_round"],
                },
            }
        else:
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            window = history["telemetry"][
                event_round:event_round + response_window_length
            ]
            post = history["telemetry"][event_round + response_window_length:]
            exact = [int(row["round"]) for row in window if action_set(row) == expected]
            extra = [int(row["round"]) for row in window if action_set(row) - expected]
            calls = [row for row in history["telemetry"] if row.get("llm_called")]
            result = {
                "case": case_id,
                "seed": seed,
                "expected": sorted(expected),
                "first_exact_round": exact[0] if exact else None,
                "success_at_10": bool(exact) and not extra,
                "exact_decisions_at_10": len(exact),
                "extra_action_rounds_at_10": extra,
                "correct_rounds_50_79": sum(
                    action_set(row) == expected for row in post
                ),
                "round_79_actions": sorted(action_set(history["telemetry"][-1])),
                "api_calls": len(calls),
                "prompt_tokens": sum(
                    int(row.get("llm_prompt_tokens", 0)) for row in calls
                ),
                "completion_tokens": sum(
                    int(row.get("llm_completion_tokens", 0)) for row in calls
                ),
                "mean_latency_ms": round(
                    sum(float(row.get("llm_latency_ms", 0.0)) for row in calls) /
                    max(1, len(calls)), 3
                ),
                "interface_error_rounds": [
                    int(row["round"]) for row in history["telemetry"]
                    if row.get("agent_interface_error") or row.get("llm_api_error") or
                    row.get("llm_parse_error") or row.get("llm_budget_error")
                ],
                "terminal_failure": None,
            }
        results.append(result)
        _write_json(out / "summary.json", {
            "status": "running", "candidate_only": True,
            "model": model["model"],
            "reasoning_effort": reasoning_effort,
            "response_window_rounds": response_window_length,
            "completed": len(results),
            "total": total, "results": results,
        })
        print(
            f"QWEN9_PROGRESS effort={reasoning_effort} "
            f"completed={len(results)}/{total} "
            f"case={case_id} seed={seed} calls={len(calls)}"
            f" terminal={result['terminal_failure'] is not None}",
            flush=True,
        )

    _write_json(out / "summary.json", {
        "status": "completed", "candidate_only": True,
        "model": model["model"], "reasoning_effort": reasoning_effort,
        "response_window_rounds": response_window_length,
        "completed": total, "total": total,
        "success_at_10": sum(row["success_at_10"] for row in results),
        "terminal_failure_count": sum(
            row["terminal_failure"] is not None for row in results
        ),
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in results),
        "total_completion_tokens": sum(row["completion_tokens"] for row in results),
        "results": results,
    })
    print(
        f"QWEN9_LOCAL_OK effort={reasoning_effort} completed={total}/{total}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ablation-group", choices=("G1", "G2", "G3", "G4", "G5"))
    parser.add_argument(
        "--reasoning-efforts", nargs="+",
        choices=("none", "low", "medium", "xhigh"), default=("low",),
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="CASE:SEED",
        help="skip one completed case/seed pair; may be repeated",
    )
    args = parser.parse_args()
    excluded = set(args.exclude)
    known = {f"{case}:{seed}" for seed in args.seeds for case in args.cases}
    unknown = excluded - known
    if unknown:
        parser.error(f"unknown --exclude pair(s): {', '.join(sorted(unknown))}")
    jobs = [
        (seed, case) for seed in args.seeds for case in args.cases
        if f"{case}:{seed}" not in excluded
    ]
    if args.check:
        assert jobs and len(set(args.reasoning_efforts)) == len(args.reasoning_efforts)
        assert set(B3_VISIBLE_ACTIONS) >= {"no_op", "dropout_handle", "robust"}
        assert XHIGH_TIMEOUT_SECONDS == 600.0
        print(
            f"QWEN9_LOCAL_CHECK_OK efforts={','.join(args.reasoning_efforts)} "
            f"jobs_per_effort={len(jobs)} total_jobs={len(jobs) * len(args.reasoning_efforts)} "
            f"rounds=80 max_calls={len(jobs) * len(args.reasoning_efforts) * 80} "
            f"xhigh_timeout_seconds={XHIGH_TIMEOUT_SECONDS:g}"
        )
        return
    for reasoning_effort in args.reasoning_efforts:
        run_effort(args, reasoning_effort, jobs)


if __name__ == "__main__":
    main()
