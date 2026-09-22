"""Run one frozen-protocol Composite Rule telemetry ablation episode."""

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents import CompositeDiagnoseAgent  # noqa: E402
from b3_composite import _write_json, prepare_episode  # noqa: E402
from b3_fingerprint import run_fingerprint, source_sha256  # noqa: E402
from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256  # noqa: E402
from llm_backend import B3_SYSTEM_PROMPT  # noqa: E402
from scripts.run_atomic_llm import (  # noqa: E402
    ATOMIC_CASES,
    FROZEN_MANIFEST_SHA,
    action_set,
    atomic_manifest,
)
from scripts.run_b3_formal_eval import _run_or_record_failure, _terminal_failure  # noqa: E402


COMPOSITE_CASES = ("real_dropout", "staggered")


def run(args):
    base = load_manifest(DEFAULT_MANIFEST)
    base_sha = manifest_sha256(base)
    if base_sha != FROZEN_MANIFEST_SHA:
        raise RuntimeError(
            f"formal Rule ablation requires frozen manifest {FROZEN_MANIFEST_SHA}; got {base_sha}"
        )

    atomic = args.case in ATOMIC_CASES
    case_id = f"atomic_{args.case}" if atomic else args.case
    manifest_path = DEFAULT_MANIFEST
    if atomic:
        manifest_path = atomic_manifest(args.out_dir / "private" / "atomic_manifest.json")
    episode_dir = args.out_dir / f"{case_id}_s{args.seed}"
    plan_path = episode_dir / "private" / "private_plans" / f"{case_id}_s{args.seed}.json"
    cfg, data, plan = prepare_episode(
        case_id, args.seed, False, False, manifest_path=manifest_path,
        plan_path=plan_path, resume=plan_path.exists(),
    )
    manifest = load_manifest(manifest_path)
    implementation_sha = source_sha256()
    agent = CompositeDiagnoseAgent(ablation_group=args.ablation_group)
    agent.name = f"composite_rule_minus_{args.ablation_group.lower()}"
    spec = {
        "name": agent.name,
        "agent_version": implementation_sha,
        "model": "deterministic",
        "ablation_group": args.ablation_group,
        "policy": {"decision_cadence": int(cfg.decide_every)},
    }
    checkpoint = (
        episode_dir / "private" / "checkpoints" / plan["episode_id"] /
        f"{agent.name}.json"
    )
    fingerprint = run_fingerprint(
        manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg, data=data,
        agent=agent, synthetic=False, smoke=False,
        prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    )
    history = _run_or_record_failure(
        checkpoint, agent, cfg, data, plan, fingerprint, spec,
        "ABLATION_RULE", "icassp-ablation-rule-v1",
    )
    expected = set(plan["hidden_event"]["canonical_bundle"])
    result = {
        "case": args.case,
        "case_id": case_id,
        "seed": args.seed,
        "ablation_group": args.ablation_group,
        "expected": sorted(expected),
        "rounds": int(cfg.rounds),
    }
    if history is None:
        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
        result.update({
            "status": "terminal_failure",
            "terminal_failure": None if terminal is None else terminal["integrity"],
        })
    else:
        event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
        window = history["telemetry"][event_round:event_round + 10]
        post = history["telemetry"][event_round + 10:]
        exact = [int(row["round"]) for row in window if action_set(row) == expected]
        extra = [int(row["round"]) for row in window if action_set(row) - expected]
        result.update({
            "status": "completed",
            "first_exact_round": exact[0] if exact else None,
            "success_at_10": bool(exact) and not extra,
            "exact_decisions_at_10": len(exact),
            "extra_action_rounds_at_10": extra,
            "correct_rounds_after_window": sum(
                action_set(row) == expected for row in post
            ),
            "final_actions": sorted(action_set(history["telemetry"][-1])),
        })
    _write_json(episode_dir / "summary.json", result)
    print(
        f"ABLATION_RULE_OK group={args.ablation_group} case={args.case} "
        f"seed={args.seed} status={result['status']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-group", choices=("G1", "G2", "G3", "G4", "G5"), required=True)
    parser.add_argument("--case", choices=tuple(ATOMIC_CASES) + COMPOSITE_CASES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 44, 56), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ablation_rule")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        assert args.case in ATOMIC_CASES or args.case in COMPOSITE_CASES
        print(
            f"ABLATION_RULE_CHECK_OK group={args.ablation_group} "
            f"case={args.case} seed={args.seed}"
        )
        return
    run(args)


if __name__ == "__main__":
    main()
