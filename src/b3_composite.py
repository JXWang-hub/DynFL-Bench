"""Thin B3 adapter from frozen episode plans to the shared Engine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parent

from agent_decision_audit import (
    DECISION_FIELDS, EVENT_FIELDS, INTEGRITY_FIELDS, SUMMARY_FIELDS,
    audit_episode, agent_config_id, make_run_id, summarize_agents,
)
from agents import (
    AdaptAgent, AutoSpawnConceptAgent, B3_VISIBLE_ACTIONS,
    CompositeDiagnoseAgent, CompositeOracleAgent, FriendSubstituteAgent,
    HandleDropoutAgent, InputShiftAgent, LabelPriorAdaptAgent, NoOpAgent,
    LLMAgent, RejectAgent, SelectClientsAgent, offline_query_fn,
)
from b3_episode_plan import (
    bind_partition, build_episode_plan, load_private_plan, resolve_episode_clients,
    save_private_plan,
)
from b3_fingerprint import git_identity, run_fingerprint, source_sha256
from b3_manifest import (
    DEFAULT_MANIFEST, load_manifest, manifest_sha256,
    partition_artifact_path, require_formal_manifest, require_versioned_output,
    response_window,
)
from config import PARTITION_MODES, Config
from engine import Engine
from evidence import ACTION_MECHANISM_SIGNAL
from llm_backend import B3_SYSTEM_PROMPT, load_model_specs, make_openai_query_fn
from run import build_data, set_seed


class RetryableRunError(RuntimeError):
    pass


def prepare_episode(case_id: str, seed: int, synthetic: bool, smoke: bool,
                    manifest_path: Path = DEFAULT_MANIFEST,
                    plan_path: Path | None = None, resume: bool = False,
                    partition_mode: str = "noniid"):
    manifest = load_manifest(manifest_path)
    if plan_path and plan_path.exists():
        plan = load_private_plan(
            plan_path, case_id, seed, 20, manifest_path, partition_mode
        )
    elif plan_path and resume:
        raise FileNotFoundError(f"private episode plan is required for resume: {plan_path}")
    else:
        plan = build_episode_plan(
            case_id, seed, manifest_path=manifest_path, partition_mode=partition_mode
        )
    hidden = plan["hidden_event"]
    event_round = int(hidden["event_schedule"][0]["round"])

    cfg = Config()
    cfg.scenario = ("staggered" if hidden["active_causes"] == ["staggered_concepts"]
                    else "b3")
    cfg.seed = seed
    cfg.dataset = "synthetic" if synthetic else manifest["protocol"]["dataset"]
    cfg.partition_mode = partition_mode
    cfg.rounds = int(manifest["protocol"]["rounds"])
    cfg.drift_round = event_round
    cfg.drift_fraction = 0.0
    cfg.formal_evidence = True
    cfg.proxy_evidence = True
    cfg.b3_active_causes = tuple(hidden["active_causes"])
    for values in hidden["event_schedule"][0]["severity"].values():
        for key, value in values.items():
            setattr(cfg, key, value)
    if "real_drift" in cfg.b3_active_causes:
        cfg.drift_type = "real"
    elif "virtual_drift" in cfg.b3_active_causes:
        cfg.drift_type = "virtual"
    elif "label_prior_drift" in cfg.b3_active_causes:
        cfg.drift_type = "label"
    if smoke:
        cfg.rounds = max(event_round + 2, response_window(manifest, event_round)[1] + 1)
        cfg.samples_per_client = 32 if synthetic else 80
        cfg.test_size = 200 if synthetic else 500
        cfg.local_epochs = 1
        cfg.batch_size = 32
        if cfg.scenario == "staggered":
            cfg.stagger_gap = 2
            cfg.samples_per_client = 80
            cfg.rounds = max(cfg.rounds, event_round + 25)
    if not torch.cuda.is_available():
        cfg.device = "cpu"

    set_seed(seed)
    data = build_data(cfg)
    dominant_labels = {
        ci: int(torch.bincount(labels, minlength=data["n_classes"]).argmax())
        for ci, labels in enumerate(data["client_y"])
    }
    planned_clients = None
    if "partial_participation_hetero" in cfg.b3_active_causes:
        k = max(1, int(cfg.participation * cfg.num_clients))
        rng = np.random.default_rng(cfg.seed + 5000 + event_round)
        planned_clients = sorted(rng.choice(cfg.num_clients, k, replace=False).tolist())
    resolved = resolve_episode_clients(
        plan, dominant_labels=dominant_labels, planned_clients=planned_clients,
        n_classes=data["n_classes"], fdms_groups=data.get("fdms_groups"),
    )
    resolved = bind_partition(
        resolved, data["partition_sha256"], partition_mode=partition_mode
    )
    if plan_path:
        save_private_plan(plan_path, resolved)
    targets = resolved["hidden_event"]["affected_clients"]
    data["dropped_clients"] = set(targets.get("dropout", {}).get("clients", ()))
    data["fault_clients"] = set(targets.get("fault", {}).get("clients", ()))
    data["drift_clients"] = set().union(*(
        set(targets[cause]["clients"])
        for cause in ("real_drift", "virtual_drift", "label_prior_drift")
        if cause in targets
    )) if any(cause in targets for cause in
              ("real_drift", "virtual_drift", "label_prior_drift")) else set()
    if cfg.scenario != "staggered":
        data["drift_schedule"] = {ci: event_round for ci in data["drift_clients"]}
    data["b3_episode_plan"] = resolved
    return cfg, data, resolved


def _private_plan_path(out_dir: Path, case_id: str, seed: int) -> Path:
    return out_dir / "private_plans" / f"{case_id}_s{seed}.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _evidence(hist: dict, event_round: int) -> dict:
    rows = hist["telemetry"]
    active = rows[event_round:]
    acted = rows[event_round + 1:]
    maximum = lambda key, source=active: max((int(row.get(key, 0)) for row in source), default=0)
    return {
        "data_shift_injected_clients": maximum("data_shift_injected_clients"),
        "fault_injected_clients": maximum("fault_injected_clients"),
        "dropout_removed_clients": maximum("dropout_removed_clients"),
        "dropout_handle_substitutions": maximum("dropout_handle_substitutions", acted),
        "friend_substitutions": maximum("friend_substitutions", acted),
        "robust_filtered_clients": maximum("robust_filtered_clients", acted),
        "planned_clients_count": maximum("planned_clients_count"),
        "available_clients_count": maximum("available_clients_count"),
        "participating_clients_count": maximum("participating_clients_count"),
        "feddaa_enabled": any(row.get("feddaa_enabled") for row in acted),
        "using_drift_adapt": any(row.get("using_drift_adapt") for row in acted),
        "moment_align_clients": maximum("moment_align_clients", acted),
        "moment_align_enabled": any(row.get("moment_align_enabled") for row in acted),
        "label_prior_adapt_enabled": any(row.get("label_prior_adapt_enabled") for row in acted),
        "selected_by_action": any(row.get("selected_by_action") for row in acted),
        "feddrift_num_models": maximum("feddrift_num_models", acted),
        "no_active_tool": not any(
            row.get("active_after") not in (None, "", "no_op") for row in active
        ),
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_agent_decision_audit(out_dir: Path, results: list[dict],
                                specs: dict[str, dict], metadata: dict) -> list[dict]:
    """Persist the three formal audit tables plus private evidence and integrity."""
    audit_dir = out_dir / "private" / "agent_decision_audit"
    events = [row for result in results for row in result["events"]]
    decisions = [row for result in results for row in result["decisions"]]
    integrity = [result["integrity"] for result in results]
    artifacts = {}
    for result in results:
        artifacts.update(result["artifacts"])
    summaries = summarize_agents(events, decisions, specs) if events else []
    _write_csv(audit_dir / "event_decision_audit.csv", events, EVENT_FIELDS)
    _write_csv(audit_dir / "decision_log.csv", decisions, DECISION_FIELDS)
    _write_csv(audit_dir / "agent_decision_summary.csv", summaries, SUMMARY_FIELDS)
    _write_json(audit_dir / "run_integrity.json", {
        "metadata": metadata,
        "runs": [{field: row[field] for field in INTEGRITY_FIELDS} for row in integrity],
    })
    for ref, payload in artifacts.items():
        _write_json(audit_dir / "artifacts" / f"{ref.removeprefix('sha256:')}.json", payload)
    return summaries


def _compact_history(history: dict) -> dict:
    telemetry_fields = {
        "round", "action_bundle", "declared_bundle", "decision_bundle_status",
        "raw_output", "public_observation_json", "full_public_observation_json",
        "agent_interface_error",
        "activated_bundle", "bundle_invalid", "decision_due",
        "emitted_at_round", "effective_from_round", "active_before", "active_after",
        "actions_started", "actions_stopped",
        "planned_clients_count", "available_clients_count",
        "participating_clients_count", "dropout_removed_clients",
        "data_shift_injected_clients", "fault_injected_clients",
        "robust_filtered_clients", "dropout_handle_substitutions",
        "friend_substitutions", "feddaa_enabled", "using_drift_adapt",
        "moment_align_clients", "moment_align_enabled",
        "label_prior_adapt_enabled", "planned_participation_rate",
        "selected_by_action", "feddrift_num_models",
        "feddrift_ari", "feddrift_nmi", "feddrift_learner_count_error",
        "feddrift_first_trigger_round", "feddrift_assignment_changed_clients",
        "feddrift_split_count", "feddrift_merge_count",
        "feddrift_trigger", "client_change_monitor_ready",
        "client_model_score_drop_p50", "client_model_score_drop_p90",
        "client_update_direction_dispersion", "client_loss_mean_delta",
        "online_roster_overlap", "client_score_exceedance_fraction",
        "client_score_exceedance_overlap", "client_update_group_separation",
        "availability_rate", "participation_gap", "global_acc",
        "acc_delta_3", "input_shift", "label_shift", "update_norm_ratio",
        "reasoning", "diagnosis", "llm_model", "llm_called", "llm_latency_ms",
        "llm_prompt_tokens", "llm_completion_tokens", "llm_cost_usd",
        "llm_api_attempts", "llm_request_sha256", "llm_replayed",
        "llm_parse_error", "llm_api_error", "llm_budget_error",
    }
    return {
        "acc": history["acc"],
        "hidden_event_log": history["hidden_event_log"],
        "telemetry": [
            {key: value for key, value in row.items() if key in telemetry_fields}
            for row in history["telemetry"]
        ],
    }


def _llm_call_records(history: dict, agent_spec: dict) -> list[dict]:
    input_price = agent_spec.get("input_cost_per_million")
    output_price = agent_spec.get("output_cost_per_million")
    records = []
    for row in history["telemetry"]:
        if not row.get("llm_called"):
            continue
        prompt = int(row.get("llm_prompt_tokens", 0))
        completion = int(row.get("llm_completion_tokens", 0))
        cost = ("unknown" if input_price is None or output_price is None else round(
            (prompt * float(input_price) + completion * float(output_price)) / 1_000_000,
            8,
        ))
        error_type = ("api" if row.get("llm_api_error") else
                      "parse" if row.get("llm_parse_error") else "none")
        records.append({
            "round": int(row["round"]),
            "model": row.get("llm_model", agent_spec.get("model", "n/a")),
            "prompt_sha256": agent_spec.get("prompt_sha256"),
            "request_sha256": row.get("llm_request_sha256", ""),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "latency_ms": float(row.get("llm_latency_ms", 0.0)),
            "attempts": int(row.get("llm_api_attempts", 0)),
            "replayed": bool(row.get("llm_replayed", False)),
            "input_cost_per_million": input_price,
            "output_cost_per_million": output_price,
            "cost_usd": cost,
            "error_type": error_type,
            "error": row.get("llm_api_error") or row.get("llm_parse_error") or "",
            "decision_summary": {
                "actions": row.get("action_bundle", "no_op").split("|"),
                "subtype": row.get("diagnosis", ""),
            },
        })
    return records


def _history_has_interface_failure(history: dict) -> bool:
    return any(
        row.get("agent_interface_error") or row.get("llm_api_error") or
        row.get("llm_budget_error")
        for row in history.get("telemetry", [])
    )


def _checkpoint_history(path: Path, agent, cfg, data, episode_id: str,
                        fingerprint, resume: bool,
                        checkpoint_metadata: dict | None = None,
                        phase_label: str = "B3_PHASE5",
                        audit_run_id: str | None = None) -> dict:
    fingerprint_sha = (fingerprint["sha256"] if isinstance(fingerprint, dict)
                       else fingerprint)
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; use --resume")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": data["b3_episode_plan"]["schema_version"],
            "episode_id": episode_id,
            "agent": agent.name,
            "run_fingerprint_sha256": fingerprint_sha,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Phase 5 checkpoint version/fingerprint mismatch: {path}")
        if checkpoint_metadata is not None and payload.get("agent_spec") != checkpoint_metadata:
            raise ValueError(f"Phase 6 checkpoint Agent spec mismatch: {path}")
        if not _history_has_interface_failure(payload["history"]):
            print(f"{phase_label}_RESUME episode_id={episode_id} agent={agent.name}", flush=True)
            return payload["history"]
        print(f"{phase_label}_RETRY episode_id={episode_id} agent={agent.name}", flush=True)

    query = getattr(agent, "query_fn", None)
    configure_journal = getattr(query, "configure_journal", None)
    if configure_journal is not None and checkpoint_metadata is not None:
        journal_path = path.with_suffix(".llm_journal.jsonl")
        if journal_path.exists() and not resume:
            raise FileExistsError(f"{journal_path} exists; use --resume")
        configure_journal(journal_path, {
            "run_fingerprint_sha256": fingerprint_sha,
            "agent_config_id": agent_config_id(checkpoint_metadata),
        })
    set_seed(cfg.seed)
    engine = Engine(cfg, data)
    try:
        history = _compact_history(engine.run(agent, log=False))
    except Exception as exc:
        if audit_run_id is not None:
            agent_spec = checkpoint_metadata or {
                "name": agent.name,
                "agent_version": source_sha256(),
                "model": "deterministic",
                "policy": {"decision_cadence": int(cfg.decide_every)},
            }
            failure_round = getattr(engine, "current_round", None)
            _write_json(path.with_suffix(".run_integrity.json"), {
                "schema_version": data["b3_episode_plan"]["schema_version"],
                "integrity": {
                    "run_id": audit_run_id,
                    "agent_id": agent.name,
                    "agent_config_id": agent_config_id(agent_spec),
                    "episode_id": episode_id,
                    "run_status": "INVALID",
                    "failure_type": "BENCHMARK_EXECUTION_FAILED",
                    "failure_round": "" if failure_round is None else failure_round,
                    "rerun_required": True,
                },
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        raise
    payload = {
        "schema_version": data["b3_episode_plan"]["schema_version"],
        "episode_id": episode_id,
        "agent": agent.name,
        "run_fingerprint_sha256": fingerprint_sha,
        "run_fingerprint": fingerprint if isinstance(fingerprint, dict) else None,
        "history": history,
    }
    if checkpoint_metadata is not None:
        payload["agent_spec"] = checkpoint_metadata
        payload["llm_calls"] = _llm_call_records(history, checkpoint_metadata)
    if _history_has_interface_failure(history):
        failed_path = (path.parent / "failed_attempts" /
                       f"{path.stem}_{time.time_ns()}.json")
        _write_json(failed_path, payload)
        print(
            f"{phase_label}_JOB_RETRYABLE episode_id={episode_id} agent={agent.name}",
            flush=True,
        )
        return history
    _write_json(path, payload)
    print(f"{phase_label}_JOB_OK episode_id={episode_id} agent={agent.name}", flush=True)
    return history


def _validate_mechanism(plan: dict, history: dict, event_round: int,
                        bundle=None) -> dict:
    evidence = _evidence(history, event_round)
    causes = set(plan["hidden_event"]["active_causes"])
    if causes & {"real_drift", "virtual_drift", "label_prior_drift"}:
        _require(evidence["data_shift_injected_clients"] > 0,
                 "data-shift mechanism was not observed")
    if "fault" in causes:
        _require(evidence["fault_injected_clients"] > 0,
                 "fault injector was not observed")
        _require(evidence["robust_filtered_clients"] > 0,
                 "robust filtering was not observed")
    if "dropout" in causes:
        _require(evidence["dropout_removed_clients"] > 0,
                 "dropout injector was not observed")
        _require(evidence["dropout_handle_substitutions"] > 0,
                 "dropout handling was not observed")
    if "real_drift" in causes:
        _require(evidence["feddaa_enabled"], "FedDAA mechanism was not observed")
    if "virtual_drift" in causes:
        _require(evidence["moment_align_clients"] > 0,
                 "moment alignment was not observed")
    if "label_prior_drift" in causes:
        _require(evidence["label_prior_adapt_enabled"],
                 "label-prior adaptation was not observed")
    if "partial_participation_hetero" in causes:
        _require(evidence["planned_clients_count"] == 8,
                 "planned participation count mismatch")
        _require(evidence["participating_clients_count"] <= 8,
                 "actual participation exceeded the plan")
    for action in bundle or plan["hidden_event"]["canonical_bundle"]:
        signal = ACTION_MECHANISM_SIGNAL[action]
        observed = (evidence[signal] > 1 if action == "spawn_concept_auto"
                    else bool(evidence[signal]))
        _require(observed, f"{action} mechanism signal {signal} was not observed")
    return evidence


def _phase5_roster(seed: int, event_round: int, causal_bundle) -> list:
    causal = CompositeOracleAgent(event_round, causal_bundle)
    causal.name = "causal_oracle"
    roster = [
        NoOpAgent(),
        AdaptAgent(), InputShiftAgent(), LabelPriorAdaptAgent(), RejectAgent(),
        HandleDropoutAgent(), FriendSubstituteAgent(), SelectClientsAgent(),
        AutoSpawnConceptAgent(), CompositeDiagnoseAgent(), causal,
    ]
    if len({agent.name for agent in roster}) != len(roster):
        raise ValueError("Phase 5 roster contains duplicate agent names")
    return roster


def _configured_llm_agent(model: dict, policy: dict, agent_version: str,
                          query_fn=None, ablation_group=None):
    query = (query_fn or make_openai_query_fn(
        model["model"], model["base_url"], model["api_key_env"],
        temperature=model["temperature"],
        input_cost_per_million=model["input_cost_per_million"],
        output_cost_per_million=model["output_cost_per_million"],
        system_prompt=B3_SYSTEM_PROMPT,
        timeout_seconds=policy["timeout_seconds"],
        retry_delays_seconds=policy["retry_delays_seconds"],
        reasoning_effort=model.get("reasoning_effort"),
    ))
    agent = LLMAgent(query, B3_VISIBLE_ACTIONS, max_calls=policy["max_calls"],
                     ablation_group=ablation_group)
    agent.name = model["name"]
    spec = {
        "name": model["name"], "kind": model["kind"], "model": model["model"],
        "agent_version": agent_version,
        "allowed_actions": list(B3_VISIBLE_ACTIONS),
        "decision_cadence": int(policy["decision_cadence"]),
        "timeout_seconds": float(policy["timeout_seconds"]),
        "retry_delays_seconds": list(policy["retry_delays_seconds"]),
        "max_calls": int(policy["max_calls"]),
        "temperature": model["temperature"],
        "base_url": model["base_url"], "api_key_env": model["api_key_env"],
        "prompt_sha256": hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "input_cost_per_million": model["input_cost_per_million"],
        "output_cost_per_million": model["output_cost_per_million"],
        "reasoning_effort": model.get("reasoning_effort"),
    }
    if ablation_group is not None:
        spec["ablation_group"] = ablation_group
    return agent, spec


def _formal_case(manifest: dict, case_id: str) -> dict:
    try:
        return next(row for row in manifest["formal_cases"] if row["case_id"] == case_id)
    except StopIteration as exc:
        raise ValueError(f"unknown formal case: {case_id}") from exc


def calibrate_performance_oracle(cases, seeds, synthetic: bool, smoke: bool,
                                 out_dir: Path, resume: bool,
                                 manifest_path: Path = DEFAULT_MANIFEST,
                                 partition_mode: str = "noniid") -> Path:
    manifest = load_manifest(manifest_path)
    if not synthetic and not smoke:
        require_formal_manifest(manifest)
        require_versioned_output(out_dir, manifest, partition_mode)
    if not synthetic and list(seeds) != manifest["protocol"]["calibration_seeds"]:
        raise ValueError("formal Performance Oracle calibration requires frozen calibration seeds")
    git = git_identity(require_clean=not synthetic and not smoke)
    implementation_sha = source_sha256()
    rows = []
    for case_id in cases:
        feasible = _formal_case(manifest, case_id)["feasible_bundles"]
        for seed in seeds:
            cfg, data, plan = prepare_episode(
                case_id, seed, synthetic, smoke, manifest_path,
                _private_plan_path(out_dir, case_id, seed), resume,
                partition_mode,
            )
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            for index, bundle in enumerate(feasible):
                agent = CompositeOracleAgent(event_round, bundle)
                agent.name = f"performance_candidate_{index}"
                checkpoint = (out_dir / "checkpoints" / plan["episode_id"] /
                              f"{agent.name}.json")
                fingerprint = run_fingerprint(
                    manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
                    data=data, agent=agent, synthetic=synthetic, smoke=smoke,
                    prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                )
                history = _checkpoint_history(
                    checkpoint, agent, cfg, data, plan["episode_id"],
                    fingerprint, resume,
                )
                _validate_mechanism(plan, history, event_round, bundle)
                rows.append({
                    "case_id": case_id,
                    "episode_id": plan["episode_id"],
                    "seed": seed,
                    "bundle_index": index,
                    "bundle": list(bundle),
                    "run_fingerprint_sha256": fingerprint["sha256"],
                    "post_event_mean_accuracy": statistics.fmean(
                        history["acc"][event_round:]
                    ),
                })
    selected = {}
    for case_id in cases:
        feasible = _formal_case(manifest, case_id)["feasible_bundles"]
        candidates = []
        for index, bundle in enumerate(feasible):
            utilities = [
                row["post_event_mean_accuracy"] for row in rows
                if row["case_id"] == case_id and row["bundle_index"] == index
            ]
            candidates.append({
                "bundle": list(bundle),
                "median_utility": round(float(statistics.median(utilities)), 8),
                "utilities": [round(float(value), 8) for value in utilities],
            })
        winner = max(enumerate(candidates), key=lambda item: (item[1]["median_utility"], -item[0]))
        selected[case_id] = {
            "selected_bundle": winner[1]["bundle"],
            "selected_index": winner[0],
            "candidates": candidates,
        }
    payload = {
        "schema_version": manifest["schema_version"],
        "status": "smoke" if synthetic or smoke else "frozen",
        "dataset": "synthetic" if synthetic else manifest["protocol"]["dataset"],
        "partition_mode": partition_mode,
        "smoke": smoke,
        "manifest_sha256": manifest_sha256(manifest),
        "implementation_sha256": implementation_sha,
        "source_sha256": implementation_sha,
        "seeds": list(seeds),
        "run_fingerprints": sorted({row["run_fingerprint_sha256"] for row in rows}),
        "cases": selected,
        "runner_git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "git_status_sha256": git["git_status_sha256"],
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = out_dir / "performance_oracle.json"
    _write_json(path, payload)
    print(
        f"B3_PERFORMANCE_ORACLE_OK cases={len(cases)} jobs={len(rows)} "
        f"status={payload['status']} artifact_sha256={payload['artifact_sha256']}",
        flush=True,
    )
    return path


def run_phase5_suite(cases, seeds, synthetic: bool, smoke: bool, out_dir: Path,
                     resume: bool, manifest_path: Path,
                     partition_mode: str = "noniid",
                     models_path: Path = SOURCE_ROOT / "benchmark_models.json",
                     fixture_models: bool = False) -> dict:
    manifest = load_manifest(manifest_path)
    if not synthetic and not smoke:
        require_formal_manifest(manifest)
        require_versioned_output(out_dir, manifest, partition_mode)
    git = git_identity(require_clean=not synthetic and not smoke)
    implementation_sha = source_sha256()
    model_specs = load_model_specs(models_path, require_keys=not fixture_models)
    model_config_sha = hashlib.sha256(json.dumps(
        model_specs, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    audit_run_id = make_run_id({
        "phase": "phase5", "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": implementation_sha, "partition_mode": partition_mode,
        "dataset": "synthetic" if synthetic else manifest["protocol"]["dataset"],
        "cases": list(cases), "seeds": list(seeds),
        "model_config_sha256": model_config_sha,
    })
    private_episodes = []
    audit_results, audit_specs = [], {}
    total_jobs = 0
    for seed in seeds:
        for case_id in cases:
            cfg, data, plan = prepare_episode(
                case_id, seed, synthetic, smoke, manifest_path,
                _private_plan_path(out_dir / "private", case_id, seed), resume,
                partition_mode,
            )
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            roster = _phase5_roster(
                seed, event_round, plan["hidden_event"]["canonical_bundle"],
            )
            policy = {
                **manifest["protocol"]["phase6"]["agent_policy"],
                "decision_cadence": int(cfg.decide_every),
                "max_calls": (int(cfg.rounds) + int(cfg.decide_every) - 1)
                             // int(cfg.decide_every),
            }
            llm_entries = [_configured_llm_agent(
                model, policy, implementation_sha,
                offline_query_fn if fixture_models else None,
            ) for model in model_specs]
            roster.extend(agent for agent, _ in llm_entries)
            agent_specs = {
                agent.name: {
                    "name": agent.name,
                    "agent_version": implementation_sha,
                    "model": "deterministic",
                    "policy": {"decision_cadence": int(cfg.decide_every)},
                }
                for agent in roster
            }
            agent_specs.update({agent.name: spec for agent, spec in llm_entries})
            histories = {}
            fingerprints = {}
            for agent in roster:
                checkpoint = (out_dir / "private" / "checkpoints" /
                              plan["episode_id"] / f"{agent.name}.json")
                fingerprint = run_fingerprint(
                    manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
                    data=data, agent=agent, synthetic=synthetic, smoke=smoke,
                    prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                )
                histories[agent.name] = _checkpoint_history(
                    checkpoint, agent, cfg, data, plan["episode_id"],
                    fingerprint, resume, checkpoint_metadata=agent_specs[agent.name],
                    audit_run_id=audit_run_id,
                )
                fingerprints[agent.name] = fingerprint["sha256"]
                total_jobs += 1

            for agent_name, history in histories.items():
                spec = agent_specs[agent_name]
                config_id = agent_config_id(spec)
                audit_specs[config_id] = spec
                audit_results.append(audit_episode(
                    audit_run_id, agent_name, spec, plan, history, manifest,
                ))

            hidden_logs = [history["hidden_event_log"] for history in histories.values()]
            if any(log != hidden_logs[0] for log in hidden_logs[1:]):
                raise ValueError("the same episode changed hidden targets across agents")
            causal = histories["causal_oracle"]
            evidence = _validate_mechanism(
                plan, causal, event_round, plan["hidden_event"]["canonical_bundle"]
            )
            private_episodes.append({
                "case_id": case_id,
                "seed": seed,
                "episode_plan": plan,
                "run_fingerprints": fingerprints,
                "mechanism_evidence": evidence,
            })

    metadata = {
        "schema_version": manifest["schema_version"],
        "evaluation_standard": "agent_decision_audit",
        "run_id": audit_run_id,
        "implementation_sha256": implementation_sha,
        "manifest_sha256": manifest_sha256(manifest),
        "model_config_sha256": model_config_sha,
        "models": [model["model"] for model in model_specs],
        "dataset": "synthetic" if synthetic else manifest["protocol"]["dataset"],
        "partition_mode": partition_mode,
        "smoke": smoke,
        "seeds": list(seeds),
        "episode_count": len(private_episodes),
        "executed_jobs": total_jobs,
        "runner_git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "git_status_sha256": git["git_status_sha256"],
    }
    summaries = _write_agent_decision_audit(
        out_dir, audit_results, audit_specs, metadata
    )
    invalid_audits = [result["integrity"] for result in audit_results
                      if result["integrity"]["run_status"] != "VALID"]
    if invalid_audits and not fixture_models:
        failed = ", ".join(
            f"{row['agent_id']}/{row['episode_id']}:{row['failure_type']}"
            for row in invalid_audits
        )
        raise RetryableRunError(
            f"formal Agent decision audit requires reruns: {failed}"
        )
    _write_json(out_dir / "private" / "phase5_audit.json", {
        "schema_version": manifest["schema_version"],
        "metadata": metadata,
        "episodes": private_episodes,
    })
    _write_json(out_dir / "run_metadata.json", metadata)
    print(
        f"B3_PHASE5_OK episodes={len(private_episodes)} agents={len(audit_specs)} "
        f"jobs={total_jobs} decision_audit=primary",
        flush=True,
    )
    return {"metadata": metadata, "agent_decision_summary": summaries}


def run_case(case_id: str, seed: int, synthetic: bool, smoke: bool,
             compare_noop: bool = False, check_friend: bool = False,
             manifest_path: Path = DEFAULT_MANIFEST,
             partition_mode: str = "noniid") -> dict:
    manifest = load_manifest(manifest_path)
    cfg, data, plan = prepare_episode(
        case_id, seed, synthetic, smoke, manifest_path,
        partition_mode=partition_mode,
    )
    event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
    agents = [CompositeOracleAgent(event_round, plan["hidden_event"]["canonical_bundle"])]
    if compare_noop:
        agents.append(NoOpAgent())
    if check_friend and "dropout" in plan["hidden_event"]["active_causes"]:
        bundle = ["friend_substitute" if action == "dropout_handle" else action
                  for action in plan["hidden_event"]["canonical_bundle"]]
        friend = CompositeOracleAgent(event_round, bundle)
        friend.name = "composite_friend_oracle"
        agents.append(friend)

    histories = {}
    for agent in agents:
        set_seed(seed)
        histories[agent.name] = Engine(cfg, data).run(agent, log=False)
    hidden_logs = [hist["hidden_event_log"] for hist in histories.values()]
    if any(log != hidden_logs[0] for log in hidden_logs[1:]):
        raise ValueError("the same episode changed hidden targets across agents")

    oracle = histories["composite_oracle"]
    evidence = _validate_mechanism(plan, oracle, event_round)
    causes = set(plan["hidden_event"]["active_causes"])
    if check_friend and "dropout" in causes:
        friend_evidence = _evidence(histories["composite_friend_oracle"], event_round)
        _require(friend_evidence["friend_substitutions"] > 0,
                 "friend substitution was not observed")
        if "real_drift" in causes:
            _require(friend_evidence["feddaa_enabled"],
                     "FedDAA was not observed in the friend-substitution run")
        evidence["friend_substitutions"] = friend_evidence["friend_substitutions"]

    result = {
        "case_id": case_id,
        "episode_id": plan["episode_id"],
        "plan_sha256": plan["plan_sha256"],
        "partition_mode": partition_mode,
        "agents": list(histories),
        "evidence": evidence,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["real_dropout", "real_fault"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase5", action="store_true")
    parser.add_argument("--calibrate-performance", action="store_true")
    parser.add_argument("--compare-noop", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-passes", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", type=Path, default=SOURCE_ROOT / "benchmark_models.json")
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="noniid")
    args = parser.parse_args()
    if args.retry_passes < 1:
        raise ValueError("--retry-passes must be at least 1")
    manifest = load_manifest(args.manifest)
    if args.out_dir:
        args.out_dir = partition_artifact_path(
            args.out_dir, manifest, args.partition_mode
        )
    if args.out:
        args.out = partition_artifact_path(
            args.out.parent, manifest, args.partition_mode
        ) / args.out.name
    formal = not (args.synthetic or args.smoke or args.check)
    if formal:
        require_formal_manifest(manifest)
        if args.out:
            require_versioned_output(args.out, manifest, args.partition_mode)
        if args.out_dir:
            require_versioned_output(args.out_dir, manifest, args.partition_mode)
    cases = (manifest["protocol"]["formal_case_ids"]
             if args.cases == ["all"] else args.cases)
    if args.calibrate_performance:
        if not args.out_dir:
            raise ValueError("--out-dir is required with --calibrate-performance")
        seeds = args.seeds or manifest["protocol"]["calibration_seeds"]
        calibrate_performance_oracle(
            cases, seeds, args.synthetic, args.smoke, args.out_dir, args.resume,
            args.manifest, args.partition_mode,
        )
        return 0
    if args.phase5:
        if not args.out_dir:
            raise ValueError("--out-dir is required with --phase5")
        for pass_index in range(args.retry_passes):
            try:
                run_phase5_suite(
                    cases, args.seeds or [args.seed], args.synthetic or args.check,
                    args.smoke or args.check,
                    args.out_dir, args.resume or pass_index > 0,
                    args.manifest, args.partition_mode, args.models, args.check,
                )
                break
            except RetryableRunError:
                if pass_index + 1 == args.retry_passes:
                    raise
                print(
                    f"B3_PHASE5_RETRY_PASS {pass_index + 2}/{args.retry_passes}",
                    flush=True,
                )
        return 0
    synthetic = args.synthetic or args.check
    smoke = args.smoke or args.check
    results = [run_case(
        case_id, args.seed, synthetic, smoke,
        compare_noop=args.compare_noop or args.check,
        check_friend=args.check and case_id == "real_dropout",
        manifest_path=args.manifest,
        partition_mode=args.partition_mode,
    ) for case_id in cases]
    payload = {
        "schema_version": manifest["schema_version"],
        "dataset": "synthetic" if synthetic else manifest["protocol"]["dataset"],
        "smoke": smoke,
        "partition_mode": args.partition_mode,
        "seed": args.seed,
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("B3_PHASE3B_OK " + " ".join(
        f"{row['case_id']}={json.dumps(row['evidence'], sort_keys=True)}" for row in results
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
