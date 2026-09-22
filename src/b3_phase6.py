"""Phase 6 rotating-stress runner. Synthetic/fixture with --check; no real run by default."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import torch

from agent_decision_audit import audit_episode, agent_config_id, make_run_id
from agents import B3_VISIBLE_ACTIONS, CompositeDiagnoseAgent, NoOpAgent, offline_query_fn
from action_contract import Action, ActionBundle
from b3_composite import (RetryableRunError, _checkpoint_history,
                          _configured_llm_agent, _llm_call_records,
                          _write_agent_decision_audit, _write_csv, _write_json)
from b3_episode_plan import bind_partition, plan_sha256, save_private_plan
from b3_fingerprint import git_identity, run_fingerprint, source_sha256
from b3_manifest import (DEFAULT_MANIFEST, load_manifest, manifest_sha256,
                         partition_artifact_path, require_formal_manifest,
                         require_versioned_output, validate_manifest)
from config import PARTITION_MODES, Config, validate_partition_mode
from llm_backend import B3_SYSTEM_PROMPT, load_model_specs
from run import build_data, set_seed


ROOT = Path(__file__).resolve().parent
CANONICAL_ACTION = {
    "real_drift": "drift_adapt",
    "fault": "robust",
    "dropout": "dropout_handle",
}
FORMAL_MANIFEST_SHA = "af5afd16150f4378edeace0ad00151904c7fb8c6d15e8bcebb9ee9a8e3eb540d"


def _workpoints(manifest: dict) -> dict:
    path = ROOT / manifest["baseline"]["freeze_path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["protocol"]["scenario_workpoints"]


def _ranked_clients(stress_seed: int, training_seed: int, cause: str,
                    num_clients: int) -> list[int]:
    return sorted(range(num_clients), key=lambda client: hashlib.sha256(
        f"phase6:{stress_seed}:{training_seed}:{cause}:{client}".encode("utf-8")
    ).digest())


def build_stress_plan(manifest: dict, training_seed: int,
                      partition_mode: str = "noniid", num_clients: int = 20) -> dict:
    partition_mode = validate_partition_mode(partition_mode)
    if num_clients < 2:
        raise ValueError("Phase 6 requires at least two clients")
    protocol = manifest["protocol"]["phase6"]
    segment = int(protocol["segment_rounds"])
    causes = list(protocol["causes"])
    random.Random(int(protocol["stress_seed"]) + int(training_seed)).shuffle(causes)
    starts = range(segment, segment * (2 * len(causes)), 2 * segment)
    workpoints = _workpoints(manifest)
    intervals, affected, schedule = {}, {}, []
    for cause, start in zip(causes, starts):
        if cause == "real_drift":
            clients = list(range(num_clients))
        elif cause == "fault":
            count = max(1, min(num_clients - 1, round(
                float(workpoints[cause]["byzantine_frac"]) * num_clients
            )))
            clients = _ranked_clients(protocol["stress_seed"], training_seed, cause,
                                      num_clients)[:count]
        else:
            count = max(1, min(num_clients - 1, round(
                float(workpoints[cause]["dropout_rate"]) * num_clients
            )))
            clients = _ranked_clients(protocol["stress_seed"], training_seed, cause,
                                      num_clients)[:count]
        clients = sorted(clients)
        intervals[cause] = {str(client): [[start, start + segment]] for client in clients}
        affected[cause] = {
            "resolver": "fixed_sha256_rank",
            "clients": clients,
            "start_round": start,
            "recover_round": start + segment,
        }
        schedule.append({
            "round": start,
            "activate": [cause],
            "severity": {cause: workpoints[cause]},
            "canonical_bundle": [CANONICAL_ACTION[cause]],
        })
    suite_sha = manifest_sha256(manifest)
    episode_token = hashlib.sha256(
        f"{suite_sha}:phase6:{training_seed}:{partition_mode}".encode("utf-8")
    ).hexdigest()[:32]
    plan = {
        "schema_version": manifest["schema_version"],
        "benchmark_version": manifest["benchmark_version"],
        "suite_manifest_sha256": suite_sha,
        "episode_id": "ep_" + episode_token,
        "case_id": "phase6_stress",
        "partition_mode": partition_mode,
        "workpoint_role": "phase6_stress",
        "training_seed": int(training_seed),
        "stress_seed": int(protocol["stress_seed"]),
        "num_clients": int(num_clients),
        "total_rounds": segment * (2 * len(causes) + 1),
        "hidden_event": {
            "active_causes": list(protocol["causes"]),
            "affected_clients": affected,
            "cause_intervals": intervals,
            "canonical_bundle": [CANONICAL_ACTION[cause] for cause in protocol["causes"]],
            "feasible_action_sets": {
                cause: manifest["causes"][cause]["feasible_actions"]
                for cause in protocol["causes"]
            },
            "event_schedule": schedule,
        },
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def prepare_stress_episode(manifest: dict, seed: int, partition_mode: str,
                           synthetic: bool, plan_path: Path, resume: bool):
    plan = build_stress_plan(manifest, seed, partition_mode)
    cfg = Config()
    cfg.scenario = "b3"
    cfg.seed = seed
    cfg.partition_mode = partition_mode
    cfg.dataset = "synthetic" if synthetic else manifest["protocol"]["dataset"]
    cfg.rounds = plan["total_rounds"]
    cfg.drift_round = min(row["round"] for row in plan["hidden_event"]["event_schedule"])
    cfg.drift_fraction = 0.0
    cfg.decide_every = int(manifest["protocol"]["phase6"]["agent_policy"]["decision_cadence"])
    cfg.formal_evidence = True
    cfg.proxy_evidence = True
    cfg.b3_active_causes = tuple(plan["hidden_event"]["active_causes"])
    for event in plan["hidden_event"]["event_schedule"]:
        for values in event["severity"].values():
            for key, value in values.items():
                setattr(cfg, key, value)
    if synthetic:
        cfg.samples_per_client = 16
        cfg.test_size = 100
        cfg.local_epochs = 1
        cfg.batch_size = 16
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    set_seed(seed)
    data = build_data(cfg)
    plan = bind_partition(plan, data["partition_sha256"], partition_mode)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("Phase 6 private plan version/fingerprint mismatch")
    else:
        save_private_plan(plan_path, plan)
    targets = plan["hidden_event"]["affected_clients"]
    data["drift_clients"] = set(targets["real_drift"]["clients"])
    data["fault_clients"] = set(targets["fault"]["clients"])
    data["dropped_clients"] = set(targets["dropout"]["clients"])
    data["drift_schedule"] = {
        client: targets["real_drift"]["start_round"] for client in data["drift_clients"]
    }
    data["b3_episode_plan"] = plan
    return cfg, data, plan


def _active_causes(plan: dict, round_idx: int) -> list[str]:
    active = []
    for cause, clients in plan["hidden_event"]["cause_intervals"].items():
        if any(start <= round_idx and (end is None or round_idx < end)
               for spans in clients.values() for start, end in spans):
            active.append(cause)
    return active


class StressOracleAgent:
    name = "stress_oracle"
    always_decide = True

    def __init__(self, plan: dict):
        self.plan = plan

    def decide(self, obs):
        actions = [CANONICAL_ACTION[cause]
                   for cause in _active_causes(self.plan, int(obs["round"]))]
        return ActionBundle(tuple(actions)) if actions else Action("no_op")


def _fixture_query(model_kind: str):
    calls = 0

    def query(text):
        nonlocal calls
        calls += 1
        query.last_usage = {"prompt_tokens": 100, "completion_tokens": 10}
        query.last_error = ""
        if model_kind == "closed" and calls == 1:
            raise RuntimeError("fixture API failure")
        if model_kind == "closed" and calls == 2:
            return "not-json"
        return offline_query_fn(text)

    query.model_name = f"{model_kind}_fixture"
    query.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    query.last_cost_usd = 0.0
    query.last_error = ""
    query.backend_spec = {"model": query.model_name, "fixture": True}
    return query


def _agent_spec(name: str, kind: str, model: str, policy: dict,
                input_price=None, output_price=None) -> dict:
    return {
        "name": name,
        "kind": kind,
        "model": model,
        "allowed_actions": list(B3_VISIBLE_ACTIONS),
        "decision_cadence": int(policy["decision_cadence"]),
        "timeout_seconds": float(policy["timeout_seconds"]),
        "retry_delays_seconds": list(policy["retry_delays_seconds"]),
        "max_calls": int(policy["max_calls"]),
        "prompt_sha256": hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "input_cost_per_million": input_price,
        "output_cost_per_million": output_price,
    }


def build_roster(manifest: dict, plan: dict, model_specs: list[dict],
                 agent_version: str, fixture: bool = False) -> list[tuple[object, dict]]:
    policy = manifest["protocol"]["phase6"]["agent_policy"]
    common = [
        (NoOpAgent(), _agent_spec("noop", "control", "n/a", policy)),
        (CompositeDiagnoseAgent(), _agent_spec(
            "composite_rule", "rule", "deterministic", policy
        )),
    ]
    agents = []
    for index, model in enumerate(model_specs):
        query = (_fixture_query("closed" if index == len(model_specs) - 1 else "model")
                 if fixture else None)
        agents.append(_configured_llm_agent(
            model, policy, agent_version, query
        ))
    oracle = StressOracleAgent(plan)
    oracle_entry = (oracle, _agent_spec(oracle.name, "oracle", "hidden_truth", policy))
    return common + [oracle_entry] + agents


def _validate_hidden_history(plan: dict, history: dict) -> None:
    logs = history.get("hidden_event_log", [])
    if len(logs) != 1:
        raise ValueError("Phase 6 history has no unique hidden schedule")
    row = logs[0]
    hidden = plan["hidden_event"]
    if any((row.get("plan_sha256") != plan["plan_sha256"],
            row.get("partition_sha256") != plan["partition_sha256"],
            row.get("event_schedule") != hidden["event_schedule"],
            row.get("cause_intervals") != hidden["cause_intervals"])):
        raise ValueError("Phase 6 history hidden schedule mismatch")


def _validate_action_shutdown(plan: dict, history: dict) -> None:
    rows = {int(row["round"]): row for row in history["telemetry"]}
    for cause, by_client in plan["hidden_event"]["cause_intervals"].items():
        end = next(iter(by_client.values()))[0][1]
        action = CANONICAL_ACTION[cause]
        if action not in rows[end]["active_before"].split("|"):
            raise AssertionError(f"{action} did not remain effective through its last active round")
        if action in rows[end + 1]["active_before"].split("|"):
            raise AssertionError(f"{action} did not change Engine behavior after shutdown")
    if not rows[next(iter(plan["hidden_event"]["cause_intervals"]["real_drift"].values()))[0][1]]["using_drift_adapt"]:
        raise AssertionError("drift adaptation behavior was not active before shutdown")


def _reliability(history: dict, spec: dict) -> dict:
    rows = history["telemetry"]
    calls = [row for row in rows if row.get("llm_called")]
    errors = [row for row in rows if (row.get("llm_parse_error") or
              row.get("llm_api_error") or row.get("llm_budget_error"))]
    latencies = sorted(float(row.get("llm_latency_ms", 0.0)) for row in calls)
    tokens_in = sum(int(row.get("llm_prompt_tokens", 0)) for row in calls)
    tokens_out = sum(int(row.get("llm_completion_tokens", 0)) for row in calls)
    prices = spec["input_cost_per_million"], spec["output_cost_per_million"]
    if not calls:
        cost = "n/a"
    elif None in prices:
        cost = "unknown"
    else:
        cost = round((tokens_in * float(prices[0]) + tokens_out * float(prices[1])) / 1_000_000, 8)
    return {
        "llm_calls": len(calls),
        "api_attempts": sum(int(row.get("llm_api_attempts", 0)) for row in calls),
        "replayed_calls": sum(bool(row.get("llm_replayed")) for row in calls),
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "latency_ms_mean": round(statistics.fmean(latencies), 3) if latencies else "n/a",
        "latency_ms_p95": round(latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)], 3)
        if latencies else "n/a",
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(rows), 8),
        "cost_usd": cost,
    }


def run_phase6(args) -> dict:
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid B3 manifest: " + "; ".join(errors))
    args.out_dir = partition_artifact_path(args.out_dir, manifest, args.partition_mode)
    formal = not (args.synthetic or args.check)
    if formal:
        actual_sha = manifest_sha256(manifest)
        if actual_sha != FORMAL_MANIFEST_SHA:
            raise ValueError(
                f"formal Phase 6 requires frozen manifest {FORMAL_MANIFEST_SHA}; "
                f"got {actual_sha}"
            )
        require_formal_manifest(manifest)
        require_versioned_output(args.out_dir, manifest, args.partition_mode)
    seeds = args.seeds or ([0] if args.check else manifest["protocol"]["evaluation_seeds"])
    git = git_identity(require_clean=formal)
    implementation_sha = source_sha256()
    model_specs = load_model_specs(args.models, require_keys=not args.check)
    model_config_sha = hashlib.sha256(json.dumps(
        model_specs, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    audit_run_id = make_run_id({
        "phase": "phase6", "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": implementation_sha, "partition_mode": args.partition_mode,
        "dataset": "synthetic" if args.synthetic or args.check else manifest["protocol"]["dataset"],
        "seeds": list(seeds), "model_config_sha256": model_config_sha,
    })
    private, reliability, all_calls = [], [], []
    audit_results, audit_specs = [], {}
    for seed in seeds:
        cfg, data, plan = prepare_stress_episode(
            manifest, seed, args.partition_mode, args.synthetic or args.check,
            args.out_dir / "private" / "plans" / f"stress_s{seed}.json", args.resume,
        )
        if args.check:
            unbound = build_stress_plan(manifest, seed, args.partition_mode)
            repeated = build_stress_plan(manifest, seed, args.partition_mode)
            paired = build_stress_plan(
                manifest, seed, "iid" if args.partition_mode == "noniid" else "noniid"
            )
            if unbound != repeated or unbound["hidden_event"] != paired["hidden_event"]:
                raise AssertionError("Phase 6 stress plan is not byte-stable and paired")
        histories, checkpoint_args = {}, None
        for agent, spec in build_roster(
                manifest, plan, model_specs, implementation_sha, args.check):
            fingerprint = run_fingerprint(
                manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg, data=data,
                agent=agent, synthetic=args.synthetic or args.check, smoke=args.check,
                prompt_sha=spec["prompt_sha256"],
            )
            checkpoint = (args.out_dir / "private" / "checkpoints" /
                          plan["episode_id"] / f"{agent.name}.json")
            history = _checkpoint_history(
                checkpoint, agent, cfg, data, plan["episode_id"], fingerprint,
                args.resume, checkpoint_metadata=spec, phase_label="B3_PHASE6",
                audit_run_id=audit_run_id,
            )
            if checkpoint_args is None:
                checkpoint_args = (checkpoint, agent, cfg, data, plan["episode_id"],
                                   fingerprint, spec)
            histories[agent.name] = history
            audit = audit_episode(
                audit_run_id, agent.name, spec, plan, history, manifest,
            )
            audit_results.append(audit)
            audit_specs[agent_config_id(spec)] = spec
            _validate_hidden_history(plan, history)
            rel = {"episode_id": plan["episode_id"], "seed": seed,
                   "agent": agent.name, **_reliability(history, spec)}
            reliability.append(rel)
            all_calls.extend({"episode_id": plan["episode_id"], "agent": agent.name, **row}
                             for row in _llm_call_records(history, spec))
        logs = [history["hidden_event_log"] for history in histories.values()]
        if any(log != logs[0] for log in logs[1:]):
            raise ValueError("Phase 6 Agents did not share one hidden schedule")
        if args.check:
            _validate_action_shutdown(plan, histories["stress_oracle"])
            checkpoint, agent, check_cfg, check_data, episode_id, fingerprint, spec = checkpoint_args
            bad_spec = {**spec, "timeout_seconds": spec["timeout_seconds"] + 1.0}
            try:
                _checkpoint_history(
                    checkpoint, agent, check_cfg, check_data, episode_id, fingerprint,
                    True, checkpoint_metadata=bad_spec, phase_label="B3_PHASE6",
                )
            except ValueError as exc:
                if "Agent spec mismatch" not in str(exc):
                    raise
            else:
                raise AssertionError("Phase 6 checkpoint accepted a different Agent spec")
        private.append({"episode_plan": plan, "agents": list(histories)})
    metadata = {
        "schema_version": manifest["schema_version"],
        "evaluation_standard": "agent_decision_audit",
        "run_id": audit_run_id,
        "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": implementation_sha,
        "model_config_sha256": model_config_sha,
        "models": [model["model"] for model in model_specs],
        "partition_mode": args.partition_mode,
        "dataset": ("synthetic" if args.synthetic or args.check
                    else manifest["protocol"]["dataset"]),
        "runner_git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
    }
    summaries = _write_agent_decision_audit(
        args.out_dir, audit_results, audit_specs, metadata
    )
    invalid_audits = [result["integrity"] for result in audit_results
                      if result["integrity"]["run_status"] != "VALID"]
    if invalid_audits and not args.check:
        failed = ", ".join(
            f"{row['agent_id']}/{row['episode_id']}:{row['failure_type']}"
            for row in invalid_audits
        )
        raise RetryableRunError(
            f"formal Agent decision audit requires reruns: {failed}"
        )
    _write_json(args.out_dir / "private" / "phase6_audit.json", {
        "metadata": metadata, "episodes": private, "llm_calls": all_calls,
    })
    _write_json(args.out_dir / "run_metadata.json", metadata)
    _write_csv(args.out_dir / "phase6_reliability_cost.csv", reliability, list(reliability[0]))
    if args.check:
        oracle = next(result for result in audit_results
                      if result["integrity"]["agent_id"] == "stress_oracle")
        failing_model = model_specs[-1]["name"]
        closed = next(row for row in reliability if row["agent"] == failing_model)
        oracle_events = [row for row in oracle["events"]
                         if row["transition_type"] != "INIT"]
        if not oracle_events or any(
                not row["ever_correct"] or not row["ended_correct"]
                for row in oracle_events):
            raise AssertionError("Phase 6 Oracle did not pass the primary decision audit")
        if closed["error_count"] < 2 or closed["cost_usd"] != "unknown":
            raise AssertionError("Phase 6 LLM failure/cost fallback was not exercised")
        closed_audit = next(result["integrity"] for result in audit_results
                            if result["integrity"]["agent_id"] == failing_model)
        if closed_audit["run_status"] != "INVALID":
            raise AssertionError("Phase 6 interface failure entered the formal audit")
    print(
        f"B3_PHASE6_OK episodes={len(private)} agents={len(audit_specs)} "
        f"partition_mode={args.partition_mode} decision_audit=primary",
        flush=True,
    )
    return {"metadata": metadata, "agent_decision_summary": summaries,
            "reliability_cost": reliability}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/b3_composite_v2/phase6"))
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="noniid")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-passes", type=int, default=3)
    parser.add_argument("--models", type=Path, default=ROOT / "benchmark_models.json")
    args = parser.parse_args()
    if args.retry_passes < 1:
        raise ValueError("--retry-passes must be at least 1")
    for pass_index in range(args.retry_passes):
        try:
            run_phase6(args)
            break
        except RetryableRunError:
            if pass_index + 1 == args.retry_passes:
                raise
            args.resume = True
            print(
                f"B3_PHASE6_RETRY_PASS {pass_index + 2}/{args.retry_passes}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
