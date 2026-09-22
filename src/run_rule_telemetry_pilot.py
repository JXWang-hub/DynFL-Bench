"""Targeted real-CIFAR check for the neutral-telemetry rule agents."""

import json
import os
from pathlib import Path

from agents import AutoSpawnConceptAgent, CompositeDiagnoseAgent
from b3_composite import _checkpoint_history, _write_json, prepare_episode
from b3_fingerprint import run_fingerprint
from b3_manifest import load_manifest, manifest_sha256


ROOT = Path(__file__).resolve().parent
CASES = os.environ.get(
    "RULE_PILOT_CASES", "staggered,real_fault,fault_dropout,real_hetero"
).split(",")
SEED = int(os.environ.get("RULE_PILOT_SEED", "202"))
RUN_ID = os.environ.get("RULE_PILOT_RUN_ID", f"neutral_rules_s{SEED}")
SYNTHETIC = os.environ.get("RULE_PILOT_SYNTHETIC") == "1"
FULL = os.environ.get("RULE_PILOT_FULL") == "1"
AGENTS = set(os.environ.get(
    "RULE_PILOT_AGENTS", "spawn_concept_auto,composite_rule"
).split(","))
OUT = ROOT.parent / "results" / f"rule_pilot_{RUN_ID}"


def emitted(history, start=0, stop=None):
    return {
        int(row["round"]): set(row.get("action_bundle", "no_op").split("|"))
        for row in history["telemetry"][start:stop]
    }


def extra_action_rounds(by_round, allowed):
    return [
        round_idx for round_idx, actions in by_round.items()
        if actions - {"no_op"} - set(allowed)
    ]


def main():
    manifest = load_manifest()
    manifest_sha = manifest_sha256(manifest)
    response_window_length = int(manifest["protocol"]["response_window_length"])
    results = []
    for case_id in CASES:
        plan_path = OUT / "private_plans" / f"{case_id}_s{SEED}.json"
        resume = plan_path.exists()
        cfg, data, plan = prepare_episode(
            case_id, SEED, synthetic=SYNTHETIC, smoke=not FULL,
            plan_path=plan_path, resume=resume,
        )
        event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
        expected = set(plan["hidden_event"]["canonical_bundle"])
        for agent in (AutoSpawnConceptAgent(), CompositeDiagnoseAgent()):
            if agent.name not in AGENTS:
                continue
            fingerprint = run_fingerprint(
                manifest_sha=manifest_sha, plan=plan, cfg=cfg, data=data,
                agent=agent, synthetic=SYNTHETIC, smoke=True, prompt_sha="none",
            )
            checkpoint = (
                OUT / "private" / "checkpoints" / plan["episode_id"] /
                f"{agent.name}.json"
            )
            history = _checkpoint_history(
                checkpoint, agent, cfg, data, plan["episode_id"], fingerprint,
                resume=resume, phase_label="RULE_PILOT", audit_run_id=RUN_ID,
            )
            all_actions = emitted(history)
            window = emitted(
                history, event_round, event_round + response_window_length
            )
            pre_allowed = {"select_clients"} if "select_clients" in expected else set()
            pre_extra = [
                round_idx for round_idx, actions in all_actions.items()
                if round_idx < event_round and actions - {"no_op"} - pre_allowed
            ]
            if agent.name == "spawn_concept_auto":
                false_spawn = [
                    round_idx for round_idx, actions in all_actions.items()
                    if "spawn_concept_auto" in actions and case_id != "staggered"
                ]
                detected = any("spawn_concept_auto" in actions for actions in window.values())
                passed = (detected if case_id == "staggered" else not false_spawn) and not pre_extra
                details = {"detected_in_window": detected, "false_spawn_rounds": false_spawn}
            else:
                exact = [round_idx for round_idx, actions in window.items()
                         if actions - {"no_op"} == expected]
                window_extra = extra_action_rounds(window, expected)
                passed = bool(exact) and not pre_extra and not window_extra
                details = {
                    "exact_rounds": exact,
                    "event_window_extra_rounds": window_extra,
                }
            results.append({
                "case": case_id,
                "agent": agent.name,
                "expected": sorted(expected),
                "event_round": event_round,
                "event_window": {
                    str(round_idx): sorted(actions) for round_idx, actions in window.items()
                },
                "pre_event_extra_rounds": pre_extra,
                "passed": passed,
                **details,
            })

    summary = {
        "run_id": RUN_ID,
        "seed": SEED,
        "dataset": "synthetic" if SYNTHETIC else "cifar10",
        "rounds": int(cfg.rounds),
        "cases": CASES,
        "passed": all(row["passed"] for row in results),
        "results": results,
    }
    _write_json(OUT / "summary.json", summary)
    print("RULE_PILOT_" + ("OK " if summary["passed"] else "FAILED ") +
          json.dumps(summary, sort_keys=True), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
