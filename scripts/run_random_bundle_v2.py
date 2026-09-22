"""Run the versioned random-bundle v2 baseline."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_decision_audit import agent_config_id, audit_episode, make_run_id  # noqa: E402
from agents import ACTION_FAMILY, RandomBundleV2Agent  # noqa: E402
from b3_composite import (  # noqa: E402
    _write_agent_decision_audit,
    _write_json,
    prepare_episode,
)
from b3_fingerprint import git_identity, run_fingerprint, source_sha256  # noqa: E402
from b3_manifest import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_manifest,
    manifest_sha256,
    partition_artifact_path,
    require_formal_manifest,
    require_versioned_output,
)
from llm_backend import B3_SYSTEM_PROMPT  # noqa: E402
from action_contract import B3_VISIBLE_ACTIONS  # noqa: E402
from scripts.run_b3_formal_eval import (  # noqa: E402
    _run_or_record_failure,
    _terminal_failure,
)
from scripts.run_atomic_llm import ATOMIC_CASES, atomic_manifest  # noqa: E402


TWO_CAUSE_CASES = (
    "real_dropout", "real_fault", "virtual_fault", "virtual_dropout",
    "label_dropout", "hetero_abrupt_dropout", "fault_dropout",
    "real_hetero", "virtual_hetero",
)
SINGLE_CAUSE_CASES = tuple(f"atomic_{case}" for case in ATOMIC_CASES) + ("staggered",)
SEEDS = (0, 44, 56)


def random_agent_spec(implementation_sha):
    return {
        "name": "random_bundle_v2", "agent_version": implementation_sha,
        "model": "deterministic",
        "policy": {
            "decision_cadence": 1, "intervention_probability": 0.1,
            "bundle_size_distribution": "uniform_1_3",
        },
    }


def build_random_agent(seed, p_act=0.1):
    return RandomBundleV2Agent(
        p_act=p_act, seed=seed, max_actions=3,
        actions=[name for name in B3_VISIBLE_ACTIONS if name != "no_op"],
    )


def run_group(cases, manifest_path, out_dir, partition_mode, git):
    manifest = load_manifest(manifest_path)
    jobs = [(case, seed) for case in cases for seed in SEEDS]
    implementation_sha = source_sha256()
    prompt_sha = hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    spec = random_agent_spec(implementation_sha)
    run_id = make_run_id({
        "phase": "random_bundle_v2", "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": implementation_sha, "partition_mode": partition_mode,
        "cases": list(cases), "seeds": list(SEEDS), "agent_spec": spec,
    })
    audits, results = [], []
    for index, (case, seed) in enumerate(jobs, 1):
        plan_path = out_dir / "private" / "private_plans" / f"{case}_s{seed}.json"
        cfg, data, plan = prepare_episode(
            case, seed, False, False, manifest_path, plan_path,
            plan_path.exists(), partition_mode,
        )
        agent = build_random_agent(seed)
        checkpoint = (
            out_dir / "private" / "checkpoints" / plan["episode_id"] /
            f"{agent.name}.json"
        )
        fingerprint = run_fingerprint(
            manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg, data=data,
            agent=agent, synthetic=False, smoke=False, prompt_sha=prompt_sha,
        )
        history = _run_or_record_failure(
            checkpoint, agent, cfg, data, plan, fingerprint, spec,
            "RANDOM_BUNDLE_V2", run_id,
        )
        if history is None:
            terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
            audits.append({
                "integrity": terminal["integrity"], "events": [],
                "decisions": [], "artifacts": {},
            })
            status = "terminal_failure"
        else:
            audit = audit_episode(run_id, agent.name, spec, plan, history, manifest)
            audits.append(audit)
            status = audit["integrity"]["run_status"].lower()
        results.append({"case": case, "seed": seed, "status": status})
        _write_json(out_dir / "summary.json", {
            "status": "running", "completed": index, "total": len(jobs),
            "results": results,
        })
        print(
            f"RANDOM_BUNDLE_V2_PROGRESS completed={index}/{len(jobs)} "
            f"case={case} seed={seed} status={status}", flush=True,
        )

    metadata = {
        "run_id": run_id, "implementation_sha256": implementation_sha,
        "manifest_sha256": manifest_sha256(manifest), "partition_mode": partition_mode,
        "cases": list(cases), "seeds": list(SEEDS), "executed_jobs": len(jobs),
        "runner_git_sha": git["git_sha"], "git_dirty": git["git_dirty"],
    }
    summaries = _write_agent_decision_audit(
        out_dir, audits, {agent_config_id(spec): spec}, metadata,
    )
    _write_json(out_dir / "summary.json", {
        "status": "completed", "completed": len(jobs), "total": len(jobs),
        "results": results, "agent_decision_summary": summaries,
    })
    print(f"RANDOM_BUNDLE_V2_OK jobs={len(jobs)} out_dir={out_dir}", flush=True)
    return len(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("two-cause", "single-cause"), default="two-cause")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--partition-mode", choices=("iid", "noniid"), default="noniid")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cases = TWO_CAUSE_CASES if args.suite == "two-cause" else SINGLE_CAUSE_CASES
    if args.check:
        agent = build_random_agent(7, p_act=1.0)
        bundles = [agent.decide({}) for _ in range(100)]
        sizes = {len(bundle.actions) for bundle in bundles}
        assert sizes == {1, 2, 3}
        assert all(
            len({ACTION_FAMILY[action.type] for action in bundle.actions}) == len(bundle.actions)
            for bundle in bundles
        )
        suite_specs = {
            suite: random_agent_spec("check") for suite in ("two-cause", "single-cause")
        }
        assert suite_specs["two-cause"] == suite_specs["single-cause"]
        assert len(cases) * len(SEEDS) == (27 if args.suite == "two-cause" else 21)
        print(
            f"RANDOM_BUNDLE_V2_CHECK_OK suite={args.suite} "
            f"jobs={len(cases) * len(SEEDS)} bundle_sizes=1,2,3"
        )
        return

    manifest = load_manifest(DEFAULT_MANIFEST)
    require_formal_manifest(manifest)
    default_name = ("random_bundle_v2_v1" if args.suite == "two-cause"
                    else "random_bundle_v2_single_cause_v1")
    requested_out = args.out_dir or ROOT / "results" / "b3_composite_v2" / default_name
    out_dir = partition_artifact_path(requested_out, manifest, args.partition_mode)
    require_versioned_output(out_dir, manifest, args.partition_mode)
    git = git_identity(require_clean=True)

    if args.suite == "two-cause":
        run_group(TWO_CAUSE_CASES, DEFAULT_MANIFEST, out_dir, args.partition_mode, git)
        return

    atomic_out = out_dir / "atomic"
    staggered_out = out_dir / "staggered"
    atomic_manifest_path = atomic_manifest(
        atomic_out / "private" / "atomic_manifest.json"
    )
    completed = run_group(
        SINGLE_CAUSE_CASES[:-1], atomic_manifest_path, atomic_out,
        args.partition_mode, git,
    )
    completed += run_group(
        ("staggered",), DEFAULT_MANIFEST, staggered_out,
        args.partition_mode, git,
    )
    _write_json(out_dir / "summary.json", {
        "status": "completed", "suite": "single-cause", "completed": completed,
        "total": 21, "groups": ["atomic", "staggered"],
    })
    print(f"RANDOM_BUNDLE_V2_SINGLE_OK jobs={completed} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
