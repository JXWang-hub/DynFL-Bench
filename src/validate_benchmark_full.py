"""Run the complete offline/synthetic benchmark validation suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(row: dict[str, str], key: str) -> bool:
    return row[key].lower() == "true"


def _verify(phase5: Path, phase6: Path) -> dict:
    manifest = json.loads((ROOT / "b3_suite_manifest.json").read_text(encoding="utf-8"))
    expected_cases = set(manifest["protocol"]["formal_case_ids"])

    phase5_audit = phase5 / "noniid" / "private" / "agent_decision_audit"
    phase5_integrity = json.loads(
        (phase5_audit / "run_integrity.json").read_text(encoding="utf-8")
    )["runs"]
    if not phase5_integrity or any(row["run_status"] != "VALID" for row in phase5_integrity):
        raise AssertionError("Phase 5 contains an invalid Agent run")
    phase5_agents = {row["agent_id"] for row in phase5_integrity}
    phase5_episodes = {row["episode_id"] for row in phase5_integrity}
    if len(phase5_integrity) != len(phase5_agents) * len(expected_cases):
        raise AssertionError("Phase 5 did not run every Agent on every case")

    phase5_events = [
        row for row in _read_csv(phase5_audit / "event_decision_audit.csv")
        if row["transition_type"] != "INIT"
    ]
    if {row["case_id"] for row in phase5_events} != expected_cases:
        raise AssertionError("Phase 5 event table is missing a formal case")
    for case_id in expected_cases:
        rows = [row for row in phase5_events if row["case_id"] == case_id]
        oracle = next(row for row in rows if row["agent_id"] == "causal_oracle")
        noop = next(row for row in rows if row["agent_id"] == "noop")
        if not (_truth(oracle, "ever_correct") and _truth(oracle, "ended_correct")
                and oracle["timing_status"] == "ON_TIME"):
            raise AssertionError(f"Phase 5 oracle failed {case_id}")
        if not _truth(noop, "never_correct"):
            raise AssertionError(f"Phase 5 NoOp unexpectedly passed {case_id}")
        if len({row["event_outcome_bucket"] for row in rows}) < 2:
            raise AssertionError(f"Phase 5 did not discriminate Agents on {case_id}")

    phase6_audit = phase6 / "noniid" / "private" / "agent_decision_audit"
    phase6_integrity = json.loads(
        (phase6_audit / "run_integrity.json").read_text(encoding="utf-8")
    )["runs"]
    valid = [row for row in phase6_integrity if row["run_status"] == "VALID"]
    invalid = [row for row in phase6_integrity if row["run_status"] == "INVALID"]
    if len(invalid) != 1 or invalid[0]["failure_type"] != "AGENT_INTERFACE_FAILED":
        raise AssertionError("Phase 6 fixture interface failure was not isolated")

    phase6_events = _read_csv(phase6_audit / "event_decision_audit.csv")
    oracle = [
        row for row in phase6_events
        if row["agent_id"] == "stress_oracle" and row["transition_type"] != "INIT"
    ]
    expected_transitions = ["ACTIVATE", "STABLE"] * 3
    if ([row["transition_type"] for row in oracle] != expected_transitions or
            any(not (_truth(row, "ever_correct") and _truth(row, "ended_correct"))
                for row in oracle)):
        raise AssertionError("Phase 6 oracle failed activation/recovery tracking")
    noop = [
        row for row in phase6_events
        if row["agent_id"] == "noop" and row["transition_type"] != "INIT"
    ]
    if any(not _truth(row, "never_correct") for row in noop
           if row["transition_type"] == "ACTIVATE"):
        raise AssertionError("Phase 6 NoOp unexpectedly passed an active disturbance")
    if any(not _truth(row, "ended_correct") for row in noop
           if row["transition_type"] == "STABLE"):
        raise AssertionError("Phase 6 NoOp failed a stable recovery segment")
    reliability = _read_csv(phase6 / "noniid" / "phase6_reliability_cost.csv")
    if len(reliability) != len(phase6_integrity):
        raise AssertionError("Phase 6 reliability table is incomplete")

    return {
        "phase5": {
            "cases": len(expected_cases),
            "agents": len(phase5_agents),
            "episodes": len(phase5_episodes),
            "jobs": len(phase5_integrity),
            "all_valid": True,
            "oracle_passed_all_cases": True,
            "noop_failed_all_active_cases": True,
            "agent_discrimination_passed": True,
        },
        "phase6": {
            "agents": len(phase6_integrity),
            "valid_runs": len(valid),
            "expected_invalid_fixture_runs": len(invalid),
            "activation_recovery_events": len(oracle),
            "oracle_tracking_passed": True,
            "noop_controls_passed": True,
            "reliability_rows": len(reliability),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("results/benchmark_validation"))
    args = parser.parse_args()
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    run_dir.mkdir()
    (out_root / "latest.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    log_path = run_dir / "run.log"
    status_path = run_dir / "status.json"
    started = datetime.now().astimezone().isoformat()

    steps = [
        ("manifest", [sys.executable, "b3_manifest.py"]),
        ("episode_plans", [sys.executable, "b3_episode_plan.py"]),
        ("action_engine_contract", [sys.executable, "../tests/test_b3_phase2.py"]),
        ("decision_audit", [sys.executable, "../tests/test_agent_decision_audit.py"]),
        ("llm_resilience", [sys.executable, "../tests/test_llm_resilience.py"]),
        ("phase5_all_cases", [
            sys.executable, "b3_composite.py", "--phase5", "--cases", "all",
            "--check", "--out-dir", str(run_dir / "phase5"),
        ]),
        ("phase6_activation_recovery", [
            sys.executable, "b3_phase6.py", "--check", "--seeds", "0",
            "--out-dir", str(run_dir / "phase6"),
        ]),
    ]

    def status(state: str, step: str, completed: int, error: str = "") -> None:
        _write_json(status_path, {
            "state": state,
            "step": step,
            "completed_steps": completed,
            "total_steps": len(steps) + 1,
            "pid": os.getpid(),
            "started_at": started,
            "updated_at": datetime.now().astimezone().isoformat(),
            "run_dir": str(run_dir),
            "error": error,
        })

    try:
        status("RUNNING", steps[0][0], 0)
        with log_path.open("a", encoding="utf-8") as log:
            for index, (name, command) in enumerate(steps):
                status("RUNNING", name, index)
                log.write(f"\n=== STEP {index + 1}/{len(steps) + 1}: {name} ===\n")
                log.flush()
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    text=True, check=False,
                )
                if result.returncode:
                    raise RuntimeError(f"{name} exited with code {result.returncode}")

            status("RUNNING", "verify_outputs", len(steps))
            log.write(f"\n=== STEP {len(steps) + 1}/{len(steps) + 1}: verify_outputs ===\n")
            log.flush()
            summary = _verify(run_dir / "phase5", run_dir / "phase6")
            _write_json(run_dir / "validation_summary.json", summary)
            log.write(json.dumps(summary, indent=2) + "\nBENCHMARK_VALIDATION_PASS\n")
            log.flush()
        status("PASS", "complete", len(steps) + 1)
        return 0
    except BaseException as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nBENCHMARK_VALIDATION_FAILED: {type(exc).__name__}: {exc}\n")
        status("FAILED", "failed", 0, f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
