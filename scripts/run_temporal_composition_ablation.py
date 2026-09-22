"""Run one paired sequential-cause episode for the timing ablation."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_decision_audit import audit_episode, make_run_id  # noqa: E402
from agents import CompositeDiagnoseAgent, NoOpAgent  # noqa: E402
from b3_composite import _configured_llm_agent, _write_json, prepare_episode  # noqa: E402
from b3_episode_plan import (  # noqa: E402
    bind_partition,
    build_episode_plan,
    plan_sha256,
    resolve_episode_clients,
    save_private_plan,
)
from b3_fingerprint import git_identity, run_fingerprint, source_sha256  # noqa: E402
from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256  # noqa: E402
from b3_phase6 import CANONICAL_ACTION, StressOracleAgent, _agent_spec, _reliability  # noqa: E402
from llm_backend import B3_SYSTEM_PROMPT  # noqa: E402
from scripts.run_atomic_llm import FROZEN_MANIFEST_SHA  # noqa: E402
from scripts.run_b3_formal_eval import _run_or_record_failure, _terminal_failure  # noqa: E402


CASES = {
    "real_fault": ("real_drift", "fault"),
    "real_dropout": ("real_drift", "dropout"),
    "fault_dropout": ("fault", "dropout"),
}
ORDERS = ("forward", "reverse")
FIRST_ROUND = 30
JOINT_ROUND = 40


def derive_temporal_plan(base: dict, order: str) -> dict:
    causes = CASES[base["case_id"]]
    first, second = causes if order == "forward" else tuple(reversed(causes))
    staged = copy.deepcopy(base)
    token = hashlib.sha256(
        f"{base['suite_manifest_sha256']}:temporal:{base['case_id']}:"
        f"{order}:{base['training_seed']}".encode("utf-8")
    ).hexdigest()[:32]
    staged.update({
        "episode_id": "ep_" + token,
        "case_id": f"temporal_{base['case_id']}_{order}",
        "workpoint_role": "temporal_composition_ablation",
        "temporal_base_case": base["case_id"],
        "temporal_order": [first, second],
    })
    hidden = staged["hidden_event"]
    severity = base["hidden_event"]["event_schedule"][0]["severity"]
    hidden["event_schedule"] = [
        {
            "round": FIRST_ROUND,
            "activate": [first],
            "severity": {first: severity[first]},
            "canonical_bundle": [CANONICAL_ACTION[first]],
        },
        {
            "round": JOINT_ROUND,
            "activate": [second],
            "severity": {second: severity[second]},
            "canonical_bundle": list(hidden["canonical_bundle"]),
        },
    ]
    for cause, start in ((first, FIRST_ROUND), (second, JOINT_ROUND)):
        hidden["affected_clients"][cause]["start_round"] = start
        intervals = hidden["cause_intervals"][cause]
        if any(len(spans) != 1 for spans in intervals.values()):
            raise ValueError("temporal ablation requires one persistent interval per client")
        hidden["cause_intervals"][cause] = {
            client: [[start, spans[0][1]]] for client, spans in intervals.items()
        }
    staged["plan_sha256"] = plan_sha256(staged)
    return staged


def prepare_temporal_episode(args):
    base_path = args.out_dir / "private" / "base_plans" / f"{args.case}_s{args.seed}.json"
    cfg, data, base = prepare_episode(
        args.case, args.seed, False, False, manifest_path=DEFAULT_MANIFEST,
        plan_path=base_path, resume=base_path.exists(),
    )
    plan = derive_temporal_plan(base, args.order)
    plan_path = (
        args.out_dir / "private" / "plans" /
        f"{plan['case_id']}_s{args.seed}.json"
    )
    save_private_plan(plan_path, plan)
    cfg.drift_round = FIRST_ROUND
    cfg.decide_every = 1
    data["b3_episode_plan"] = plan
    data["drift_schedule"] = {
        int(client): int(spans[0][0])
        for cause in ("real_drift", "virtual_drift", "label_prior_drift")
        for client, spans in plan["hidden_event"]["cause_intervals"].get(cause, {}).items()
    }
    return cfg, data, plan


def configured_agent(args, manifest, plan, implementation_sha):
    policy = {
        **manifest["protocol"]["phase6"]["agent_policy"],
        "decision_cadence": 1,
        "max_calls": int(plan["total_rounds"]),
    }
    if args.agent == "noop":
        agent = NoOpAgent()
        return agent, _agent_spec(agent.name, "control", "n/a", policy)
    if args.agent == "rule":
        agent = CompositeDiagnoseAgent()
        return agent, _agent_spec(agent.name, "rule", "deterministic", policy)
    if args.agent == "oracle":
        agent = StressOracleAgent(plan)
        agent.name = "temporal_oracle"
        return agent, _agent_spec(agent.name, "oracle", "hidden_truth", policy)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("missing DASHSCOPE_API_KEY")
    model = {
        "name": "qwen3p8_flash_low",
        "kind": "closed",
        "model": "qwen3.8-flash",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "temperature": 0.6,
        "reasoning_effort": "low",
        "input_cost_per_million": None,
        "output_cost_per_million": None,
    }
    return _configured_llm_agent(model, policy, implementation_sha)


def run(args):
    manifest = load_manifest(DEFAULT_MANIFEST)
    base_sha = manifest_sha256(manifest)
    if base_sha != FROZEN_MANIFEST_SHA:
        raise RuntimeError(
            f"temporal ablation requires frozen manifest {FROZEN_MANIFEST_SHA}; got {base_sha}"
        )
    git_identity(require_clean=True)
    cfg, data, plan = prepare_temporal_episode(args)
    implementation_sha = source_sha256()
    agent, spec = configured_agent(args, manifest, plan, implementation_sha)
    audit_run_id = make_run_id({
        "phase": "temporal_composition_ablation",
        "manifest_sha256": base_sha,
        "source_sha256": implementation_sha,
        "case": args.case,
        "order": args.order,
        "seed": args.seed,
        "agent": agent.name,
    })
    episode_dir = args.out_dir / plan["case_id"] / f"seed_{args.seed}"
    checkpoint = (
        episode_dir / "private" / "checkpoints" / plan["episode_id"] /
        f"{agent.name}.json"
    )
    fingerprint = run_fingerprint(
        manifest_sha=base_sha, plan=plan, cfg=cfg, data=data, agent=agent,
        synthetic=False, smoke=False,
        prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    )
    history = _run_or_record_failure(
        checkpoint, agent, cfg, data, plan, fingerprint, spec,
        "TEMPORAL_ABLATION", audit_run_id,
    )
    if history is None:
        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
        _write_json(episode_dir / "summary" / f"{agent.name}.json", {
            "status": "terminal_failure",
            "case": args.case,
            "order": args.order,
            "seed": args.seed,
            "agent": agent.name,
            "terminal_failure": None if terminal is None else terminal["integrity"],
        })
        return
    audit = audit_episode(audit_run_id, agent.name, spec, plan, history, manifest)
    _write_json(episode_dir / "private" / "audit" / f"{agent.name}.json", audit)
    events = [{
        key: event.get(key) for key in (
            "event_id", "transition_type", "event_round", "active_causes_after",
            "emit_deadline", "correct_decision_round", "decision_delay",
            "ever_correct", "ended_correct", "timing_status",
            "post_correct_regression_count", "ever_over_intervened",
        )
    } for event in audit["events"]]
    _write_json(episode_dir / "summary" / f"{agent.name}.json", {
        "status": audit["integrity"]["run_status"],
        "case": args.case,
        "order": args.order,
        "seed": args.seed,
        "agent": agent.name,
        "events": events,
        "reliability": _reliability(history, spec),
    })
    if audit["integrity"]["run_status"] != "VALID":
        raise RuntimeError(
            f"invalid temporal audit: {audit['integrity']['failure_type']}"
        )
    print(
        f"TEMPORAL_ABLATION_OK case={args.case} order={args.order} "
        f"seed={args.seed} agent={agent.name}",
        flush=True,
    )


def check():
    from agent_decision_audit import compile_events

    manifest = load_manifest(DEFAULT_MANIFEST)
    audit_manifest = copy.deepcopy(manifest)
    audit_manifest["protocol"]["response_window_length"] = 10
    labels = {client: client % 10 for client in range(20)}
    for case in CASES:
        base = build_episode_plan(
            case, 0, episode_id="ep_" + "0" * 32,
            manifest_path=DEFAULT_MANIFEST,
        )
        base = bind_partition(
            resolve_episode_clients(base, dominant_labels=labels, n_classes=10),
            "0" * 64,
        )
        original_clients = {
            cause: target["clients"]
            for cause, target in base["hidden_event"]["affected_clients"].items()
        }
        for order in ORDERS:
            staged = derive_temporal_plan(base, order)
            assert [row["round"] for row in staged["hidden_event"]["event_schedule"]] == [30, 40]
            assert original_clients == {
                cause: target["clients"]
                for cause, target in staged["hidden_event"]["affected_clients"].items()
            }
            events = compile_events(
                staged, audit_manifest,
                [{"round": round_idx, "decision_due": True} for round_idx in range(80)],
            )
            assert [event["event_round"] for event in events] == [0, 30, 40]
            assert [event["emit_deadline"] for event in events[1:]] == [39, 49]
            oracle = StressOracleAgent(staged)
            assert oracle.decide({"round": FIRST_ROUND}).action_types == (
                CANONICAL_ACTION[staged["temporal_order"][0]],
            )
            assert set(oracle.decide({"round": JOINT_ROUND}).action_types) == {
                CANONICAL_ACTION[cause] for cause in CASES[case]
            }
    print("TEMPORAL_ABLATION_CHECK_OK cases=3 orders=2 seeds=0,44,56")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 44, 56), required=True)
    parser.add_argument("--agent", choices=("noop", "rule", "oracle", "qwen"), required=True)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "temporal_composition_ablation",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    run(args)


if __name__ == "__main__":
    main()
