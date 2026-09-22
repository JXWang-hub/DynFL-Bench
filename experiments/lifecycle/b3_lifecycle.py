"""Lifecycle extensions for the frozen B3 Phase 6 protocol."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from agent_decision_audit import audit_episode, agent_config_id, make_run_id
from b3_composite import (RetryableRunError, _checkpoint_history,
                          _llm_call_records, _write_agent_decision_audit,
                          _write_csv, _write_json)
from b3_episode_plan import bind_partition, plan_sha256, save_private_plan
from b3_fingerprint import git_identity, run_fingerprint, source_sha256
from b3_manifest import (DEFAULT_MANIFEST, load_manifest, manifest_sha256,
                         partition_artifact_path, require_formal_manifest,
                         require_versioned_output, validate_manifest)
from b3_phase6 import (CANONICAL_ACTION, FORMAL_MANIFEST_SHA, _reliability,
                       _workpoints, _ranked_clients, build_roster)
from config import PARTITION_MODES, Config, validate_partition_mode
from llm_backend import load_model_specs
from run import build_data, set_seed


CAUSES = ("real_drift", "fault", "dropout")


def _clients(manifest: dict, seed: int, cause: str, num_clients: int) -> list[int]:
    if cause == "real_drift":
        return list(range(num_clients))
    workpoint = _workpoints(manifest)[cause]
    key = "byzantine_frac" if cause == "fault" else "dropout_rate"
    count = max(1, min(num_clients - 1, round(float(workpoint[key]) * num_clients)))
    return sorted(_ranked_clients(manifest["protocol"]["phase6"]["stress_seed"],
                                  seed, cause, num_clients)[:count])


def _intervals(states: list[tuple[str, ...]], segment: int,
               causes: tuple[str, ...]) -> dict[str, dict[str, list[list[int]]]]:
    result = {cause: {} for cause in causes}
    for cause in causes:
        spans = []
        start = None
        for index, state in enumerate(states + [()]):
            active = cause in state
            if active and start is None:
                start = index * segment
            if start is not None and not active:
                spans.append([start, index * segment])
                start = None
        result[cause] = spans
    return result


def _build_plan(manifest: dict, seed: int, partition_mode: str, mode: str,
                label: str, states: list[tuple[str, ...]], repeat_id: int = 0,
                num_clients: int = 20) -> dict:
    partition_mode = validate_partition_mode(partition_mode)
    segment = int(manifest["protocol"]["phase6"]["segment_rounds"])
    causes = tuple(dict.fromkeys(cause for state in states for cause in state))
    spans = _intervals(states, segment, causes)
    workpoints = _workpoints(manifest)
    affected, intervals = {}, {}
    for cause in causes:
        clients = _clients(manifest, seed, cause, num_clients)
        intervals[cause] = {str(client): spans[cause] for client in clients}
        affected[cause] = {
            "resolver": "fixed_sha256_rank",
            "clients": clients,
            "start_round": spans[cause][0][0],
            "recover_round": spans[cause][-1][1],
        }
    schedule = []
    previous = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state == previous:
            continue
        round_idx = index * segment
        schedule.append({
            "round": round_idx,
            "activate": [cause for cause in state if cause not in previous],
            "severity": {cause: workpoints[cause] for cause in state},
            "canonical_bundle": [CANONICAL_ACTION[cause] for cause in state] or ["no_op"],
            "emit_deadline": round_idx + segment - 1,
        })
        previous = state
    suite_sha = manifest_sha256(manifest)
    token = hashlib.sha256(
        f"{suite_sha}:lifecycle:{mode}:{label}:{seed}:{partition_mode}:{repeat_id}".encode()
    ).hexdigest()[:32]
    plan = {
        "schema_version": manifest["schema_version"],
        "benchmark_version": manifest["benchmark_version"],
        "suite_manifest_sha256": suite_sha,
        "episode_id": "ep_" + token,
        "case_id": f"lifecycle_{mode}",
        "partition_mode": partition_mode,
        "workpoint_role": f"lifecycle_{mode}",
        "training_seed": int(seed),
        "stress_seed": int(manifest["protocol"]["phase6"]["stress_seed"]),
        "num_clients": int(num_clients),
        "total_rounds": len(states) * segment,
        "lifecycle_mode": mode,
        "lifecycle_label": label,
        "api_repeat_id": int(repeat_id),
        "hidden_event": {
            "active_causes": list(causes),
            "affected_clients": affected,
            "cause_intervals": intervals,
            "canonical_bundle": [CANONICAL_ACTION[cause] for cause in causes],
            "feasible_action_sets": {
                cause: manifest["causes"][cause]["feasible_actions"] for cause in causes
            },
            "event_schedule": schedule,
        },
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def build_lifecycle_plan(manifest: dict, seed: int, order: tuple[str, ...],
                         partition_mode="noniid", repeat_id=0) -> dict:
    if set(order) != set(CAUSES) or len(order) != len(CAUSES):
        raise ValueError("lifecycle order must be a permutation of real_drift,fault,dropout")
    first, second, third = order
    states = [(), (first,), (first, second), order, (second, third), (third,), ()]
    return _build_plan(manifest, seed, partition_mode, "three_cause", "-".join(order),
                       states, repeat_id)


def build_recurrence_plan(manifest: dict, seed: int, cause: str,
                          partition_mode="noniid") -> dict:
    if cause not in CAUSES:
        raise ValueError(f"unknown recurrence cause {cause!r}")
    return _build_plan(manifest, seed, partition_mode, "recurrence", cause,
                       [(), (cause,), (), (cause,), ()])


def prepare_episode(manifest: dict, plan: dict, synthetic: bool, plan_path: Path,
                    resume: bool):
    cfg = Config()
    cfg.scenario = "b3"
    cfg.seed = int(plan["training_seed"])
    cfg.partition_mode = plan["partition_mode"]
    cfg.dataset = "synthetic" if synthetic else manifest["protocol"]["dataset"]
    cfg.rounds = int(plan["total_rounds"])
    cfg.drift_round = min(row["round"] for row in plan["hidden_event"]["event_schedule"])
    cfg.drift_fraction = 0.0
    cfg.decide_every = int(manifest["protocol"]["phase6"]["agent_policy"]["decision_cadence"])
    cfg.formal_evidence = True
    cfg.proxy_evidence = True
    cfg.b3_active_causes = tuple(plan["hidden_event"]["active_causes"])
    for row in plan["hidden_event"]["event_schedule"]:
        for values in row["severity"].values():
            for key, value in values.items():
                setattr(cfg, key, value)
    if synthetic:
        cfg.samples_per_client, cfg.test_size = 16, 100
        cfg.local_epochs, cfg.batch_size = 1, 16
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    set_seed(cfg.seed)
    data = build_data(cfg)
    plan = bind_partition(plan, data["partition_sha256"], cfg.partition_mode)
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("lifecycle private plan version/fingerprint mismatch")
    else:
        save_private_plan(plan_path, plan)
    targets = plan["hidden_event"]["affected_clients"]
    data["drift_clients"] = set(targets.get("real_drift", {}).get("clients", ()))
    data["fault_clients"] = set(targets.get("fault", {}).get("clients", ()))
    data["dropped_clients"] = set(targets.get("dropout", {}).get("clients", ()))
    data["drift_schedule"] = {
        client: targets["real_drift"]["start_round"] for client in data["drift_clients"]
    }
    data["b3_episode_plan"] = plan
    return cfg, data, plan


def _specs(args, manifest: dict, seeds: list[int]):
    if args.protocol == "three_cause":
        orders = list(itertools.permutations(CAUSES))
        if args.api_repeat_id:
            return [(order[index % len(seeds)], build_lifecycle_plan(
                manifest, seeds[index % len(seeds)], order, args.partition_mode,
                args.api_repeat_id)) for index, order in enumerate(orders)]
        return [(order, build_lifecycle_plan(manifest, seed, order, args.partition_mode))
                for seed in seeds for order in orders]
    return [(cause, build_recurrence_plan(manifest, seed, cause, args.partition_mode))
            for seed in seeds for cause in CAUSES]


def _select_roster(roster, roles: tuple[str, ...]):
    selected = []
    for agent, spec in roster:
        role = ("noop" if spec["name"] == "noop" else
                "rule" if spec["name"] == "composite_rule" else
                "oracle" if spec["name"] == "stress_oracle" else "qwen")
        if role in roles:
            selected.append((agent, spec))
    return selected


def _validate_oracle(plan: dict, history: dict) -> None:
    rows = {int(row["round"]): row for row in history["telemetry"]}
    for cause, by_client in plan["hidden_event"]["cause_intervals"].items():
        action = CANONICAL_ACTION[cause]
        for start, end in next(iter(by_client.values())):
            if action not in rows[end]["active_before"].split("|"):
                raise AssertionError(f"{cause} action did not persist through round {end - 1}")
            if end + 1 in rows and action in rows[end + 1]["active_before"].split("|"):
                raise AssertionError(f"{cause} action remained active after round {end}")


def _check_plans(manifest: dict, partition_mode: str) -> None:
    for order in itertools.permutations(CAUSES):
        plan = build_lifecycle_plan(manifest, 0, order, partition_mode)
        events = plan["hidden_event"]["event_schedule"]
        assert [row["round"] for row in events] == [10, 20, 30, 40, 50, 60]
        assert events[2]["canonical_bundle"] == [CANONICAL_ACTION[cause] for cause in order]
        assert plan == build_lifecycle_plan(manifest, 0, order, partition_mode)
    for cause in CAUSES:
        plan = build_recurrence_plan(manifest, 0, cause, partition_mode)
        spans = next(iter(plan["hidden_event"]["cause_intervals"][cause].values()))
        assert spans == [[10, 20], [30, 40]]
    print("B3_LIFECYCLE_PLAN_CHECK_OK orders=6 recurrence_causes=3")


def run(args) -> dict:
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid B3 manifest: " + "; ".join(errors))
    args.out_dir = partition_artifact_path(args.out_dir, manifest, args.partition_mode)
    formal = not (args.synthetic or args.check)
    if formal:
        if manifest_sha256(manifest) != FORMAL_MANIFEST_SHA:
            raise ValueError("formal lifecycle run requires the frozen B3 manifest")
        require_formal_manifest(manifest)
        require_versioned_output(args.out_dir, manifest, args.partition_mode)
    seeds = args.seeds or ([0] if args.check else manifest["protocol"]["evaluation_seeds"])
    _check_plans(manifest, args.partition_mode)
    specs = _specs(args, manifest, seeds)
    if args.check:
        specs = specs[:1]
    git = git_identity(require_clean=formal)
    implementation_sha = source_sha256()
    model_specs = load_model_specs(args.models, require_keys=not args.check)
    model_config_sha = hashlib.sha256(json.dumps(model_specs, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
    audit_run_id = make_run_id({
        "phase": "lifecycle", "protocol": args.protocol, "repeat": args.api_repeat_id,
        "manifest_sha256": manifest_sha256(manifest), "source_sha256": implementation_sha,
        "partition_mode": args.partition_mode, "seeds": seeds,
        "model_config_sha256": model_config_sha, "roles": args.agents,
    })
    audits, audit_specs, private, reliability, calls = [], {}, [], [], []
    for label, unbound_plan in specs:
        plan_path = args.out_dir / "private" / "plans" / f"{unbound_plan['episode_id']}.json"
        cfg, data, plan = prepare_episode(manifest, unbound_plan, args.synthetic or args.check,
                                          plan_path, args.resume)
        histories = {}
        roster = _select_roster(build_roster(manifest, plan, model_specs, implementation_sha,
                                             args.check), tuple(args.agents))
        if not roster:
            raise ValueError("agent selection produced an empty roster")
        for agent, spec in roster:
            fingerprint = run_fingerprint(manifest_sha=manifest_sha256(manifest), plan=plan,
                                          cfg=cfg, data=data, agent=agent,
                                          synthetic=args.synthetic or args.check,
                                          smoke=args.check, prompt_sha=spec["prompt_sha256"])
            checkpoint = args.out_dir / "private" / "checkpoints" / plan["episode_id"] / f"{agent.name}.json"
            history = _checkpoint_history(checkpoint, agent, cfg, data, plan["episode_id"], fingerprint,
                                          args.resume, checkpoint_metadata=spec,
                                          phase_label="B3_LIFECYCLE", audit_run_id=audit_run_id)
            histories[agent.name] = history
            audit = audit_episode(audit_run_id, agent.name, spec, plan, history, manifest)
            audits.append(audit)
            audit_specs[agent_config_id(spec)] = spec
            rel = {"episode_id": plan["episode_id"], "seed": plan["training_seed"],
                   "lifecycle_label": str(label), "api_repeat_id": plan["api_repeat_id"],
                   "agent": agent.name, **_reliability(history, spec)}
            reliability.append(rel)
            calls.extend({"episode_id": plan["episode_id"], "agent": agent.name, **row}
                         for row in _llm_call_records(history, spec))
        if len({json.dumps(history["hidden_event_log"], sort_keys=True) for history in histories.values()}) != 1:
            raise ValueError("lifecycle agents did not share one hidden plan")
        if args.check and "stress_oracle" in histories:
            _validate_oracle(plan, histories["stress_oracle"])
        private.append({"episode_plan": plan, "agents": list(histories)})
    metadata = {
        "schema_version": manifest["schema_version"], "evaluation_standard": "agent_decision_audit",
        "run_id": audit_run_id, "protocol": args.protocol, "api_repeat_id": args.api_repeat_id,
        "manifest_sha256": manifest_sha256(manifest), "source_sha256": implementation_sha,
        "model_config_sha256": model_config_sha, "models": [row["model"] for row in model_specs],
        "partition_mode": args.partition_mode,
        "dataset": "synthetic" if args.synthetic or args.check else manifest["protocol"]["dataset"],
        "runner_git_sha": git["git_sha"], "git_dirty": git["git_dirty"],
    }
    summaries = _write_agent_decision_audit(args.out_dir, audits, audit_specs, metadata)
    invalid = [row["integrity"] for row in audits if row["integrity"]["run_status"] != "VALID"]
    if invalid and formal:
        raise RetryableRunError("formal lifecycle audit requires reruns")
    _write_json(args.out_dir / "private" / "lifecycle_audit.json", {
        "metadata": metadata, "episodes": private, "llm_calls": calls,
    })
    _write_json(args.out_dir / "run_metadata.json", metadata)
    _write_csv(args.out_dir / "lifecycle_reliability_cost.csv", reliability, list(reliability[0]))
    print(f"B3_LIFECYCLE_OK protocol={args.protocol} episodes={len(private)} agents={len(audit_specs)}")
    return {"metadata": metadata, "agent_decision_summary": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("results/b3_composite_v2/lifecycle"))
    parser.add_argument("--protocol", choices=("three_cause", "recurrence"), required=True)
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="noniid")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--agents", nargs="+", choices=("noop", "rule", "oracle", "qwen"),
                        default=("noop", "rule", "oracle", "qwen"))
    parser.add_argument("--api-repeat-id", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-passes", type=int, default=3)
    parser.add_argument("--models", type=Path, default=SRC / "benchmark_models.json")
    args = parser.parse_args()
    if args.api_repeat_id and (args.protocol != "three_cause" or tuple(args.agents) != ("qwen",)):
        parser.error("--api-repeat-id requires --protocol three_cause --agents qwen")
    if args.retry_passes < 1:
        parser.error("--retry-passes must be at least 1")
    for attempt in range(args.retry_passes):
        try:
            run(args)
            return 0
        except RetryableRunError:
            if attempt + 1 == args.retry_passes:
                raise
            args.resume = True
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
