"""Run the frozen 10-case evaluation: deterministic agents first, then two LLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_decision_audit import agent_config_id, audit_episode, make_run_id  # noqa: E402
from b3_composite import (  # noqa: E402
    RetryableRunError,
    _checkpoint_history,
    _configured_llm_agent,
    _phase5_roster,
    _validate_mechanism,
    _write_agent_decision_audit,
    _write_json as _write_b3_json,
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


LOW_REASONING_MODELS = {
    "models": [
        {
            "name": "qwen3p8_flash_low",
            "kind": "closed",
            "model": "qwen3.8-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "requires_api_key": True,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "input_cost_per_million": None,
            "output_cost_per_million": None,
        },
        {
            "name": "deepseek_v4_pro_0813_low",
            "kind": "closed",
            "model": "deepseek-v4-pro-0813",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "requires_api_key": True,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "input_cost_per_million": None,
            "output_cost_per_million": None,
        },
    ]
}


def _failure_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".run_integrity.json")


def _terminal_failure(checkpoint: Path, fingerprint_sha: str | None = None):
    path = _failure_path(checkpoint)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    integrity = payload.get("integrity", {})
    if (integrity.get("failure_type") != "NUMERICAL_DIVERGENCE" or
            integrity.get("rerun_required") is not False):
        return None
    if fingerprint_sha is not None and payload.get("run_fingerprint_sha256") != fingerprint_sha:
        return None
    return payload


def _record_numerical_failure(checkpoint: Path, fingerprint_sha: str,
                              exc: ValueError) -> dict:
    payload = json.loads(_failure_path(checkpoint).read_text(encoding="utf-8"))
    payload["run_fingerprint_sha256"] = fingerprint_sha
    payload["integrity"].update({
        "failure_type": "NUMERICAL_DIVERGENCE",
        "rerun_required": False,
    })
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
    _write_b3_json(_failure_path(checkpoint), payload)
    return payload


def _is_numerical_failure(exc: ValueError) -> bool:
    message = str(exc)
    return (
        ("public observation field" in message and "is not number" in message) or
        message == "invalid FedDAA report payload"
    )


def _run_or_record_failure(checkpoint: Path, agent, cfg, data, plan: dict,
                           fingerprint: dict, spec: dict, phase_label: str,
                           audit_run_id: str):
    fingerprint_sha = fingerprint["sha256"]
    terminal = _terminal_failure(checkpoint, fingerprint_sha)
    if terminal is not None:
        print(
            f"{phase_label}_RESUME_TERMINAL episode_id={plan['episode_id']} "
            f"agent={agent.name} round={terminal['integrity']['failure_round']}",
            flush=True,
        )
        return None
    try:
        return _checkpoint_history(
            checkpoint, agent, cfg, data, plan["episode_id"], fingerprint,
            True, checkpoint_metadata=spec,
            phase_label=phase_label, audit_run_id=audit_run_id,
        )
    except ValueError as exc:
        if not _is_numerical_failure(exc):
            raise
        terminal = _record_numerical_failure(checkpoint, fingerprint_sha, exc)
        print(
            f"{phase_label}_JOB_TERMINAL episode_id={plan['episode_id']} "
            f"agent={agent.name} failure=NUMERICAL_DIVERGENCE "
            f"round={terminal['integrity']['failure_round']}",
            flush=True,
        )
        return None


def require_deterministic_complete(out_dir: Path, cases: list[str],
                                   seeds: list[int]) -> None:
    """Refuse to start paid calls until every deterministic checkpoint exists."""
    missing: list[str] = []
    for seed in seeds:
        for case_id in cases:
            plan_path = out_dir / "private" / "private_plans" / f"{case_id}_s{seed}.json"
            if not plan_path.exists():
                missing.append(str(plan_path))
                continue
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            roster = _phase5_roster(
                seed, event_round, plan["hidden_event"]["canonical_bundle"]
            )
            checkpoint_dir = out_dir / "private" / "checkpoints" / plan["episode_id"]
            missing.extend(
                str(checkpoint_dir / f"{agent.name}.json")
                for agent in roster
                if not (checkpoint_dir / f"{agent.name}.json").exists() and
                _terminal_failure(checkpoint_dir / f"{agent.name}.json") is None
            )
    if missing:
        raise RuntimeError(
            f"API stage blocked: {len(missing)} deterministic artifacts are missing; "
            "run --stage deterministic first. First missing: " + missing[0]
        )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _self_check_terminal_failure() -> None:
    assert _is_numerical_failure(
        ValueError("public observation field 'x' is not number")
    )
    assert _is_numerical_failure(ValueError("invalid FedDAA report payload"))
    assert not _is_numerical_failure(ValueError("checkpoint mismatch"))
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "agent.json"
        _write_b3_json(_failure_path(checkpoint), {
            "integrity": {"failure_type": "BENCHMARK_EXECUTION_FAILED",
                          "failure_round": 12, "rerun_required": True},
            "error_type": "ValueError", "error": "old",
        })
        _record_numerical_failure(
            checkpoint, "fingerprint",
            ValueError("public observation field 'x' is not number"),
        )
        assert _terminal_failure(checkpoint, "fingerprint") is not None


def deterministic_stage(manifest: dict, out_dir: Path, cases: list[str],
                        seeds: list[int], partition_mode: str) -> None:
    """Populate the native Phase 5 deterministic checkpoints without an API model."""
    git_identity(require_clean=True)
    implementation_sha = source_sha256()
    prompt_sha = hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    total = len(cases) * len(seeds) * 12
    completed = 0
    for seed in seeds:
        for case_id in cases:
            plan_path = out_dir / "private" / "private_plans" / f"{case_id}_s{seed}.json"
            cfg, data, plan = prepare_episode(
                case_id, seed, False, False, DEFAULT_MANIFEST,
                plan_path, plan_path.exists(), partition_mode,
            )
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            roster = _phase5_roster(
                seed, event_round, plan["hidden_event"]["canonical_bundle"]
            )
            for agent in roster:
                spec = {
                    "name": agent.name,
                    "agent_version": implementation_sha,
                    "model": "deterministic",
                    "policy": {"decision_cadence": int(cfg.decide_every)},
                }
                checkpoint = (
                    out_dir / "private" / "checkpoints" / plan["episode_id"] /
                    f"{agent.name}.json"
                )
                fingerprint = run_fingerprint(
                    manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
                    data=data, agent=agent, synthetic=False, smoke=False,
                    prompt_sha=prompt_sha,
                )
                _run_or_record_failure(
                    checkpoint, agent, cfg, data, plan, fingerprint, spec,
                    "B3_DETERMINISTIC", "formal-deterministic-prestage",
                )
                completed += 1
                print(
                    f"B3_QUEUE_PROGRESS stage=deterministic completed={completed}/{total} "
                    f"case={case_id} seed={seed} agent={agent.name}",
                    flush=True,
                )


def api_stage(manifest: dict, out_dir: Path, cases: list[str], seeds: list[int],
              partition_mode: str, retries: int) -> None:
    require_deterministic_complete(out_dir, cases, seeds)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "deterministic stage is complete; set DASHSCOPE_API_KEY and rerun with --stage api"
        )
    models_path = out_dir / "private" / "benchmark_models_low.json"
    write_json(models_path, LOW_REASONING_MODELS)
    for attempt in range(retries):
        try:
            _api_only_pass(manifest, out_dir, cases, seeds, partition_mode)
            return
        except RetryableRunError:
            if attempt + 1 == retries:
                raise
            print(f"B3_API_RETRY_PASS {attempt + 2}/{retries}", flush=True)


def _api_only_pass(manifest: dict, out_dir: Path, cases: list[str],
                   seeds: list[int], partition_mode: str) -> None:
    """Audit saved deterministic runs and execute only the two paid agents."""
    git = git_identity(require_clean=True)
    implementation_sha = source_sha256()
    model_specs = LOW_REASONING_MODELS["models"]
    model_config_sha = hashlib.sha256(json.dumps(
        model_specs, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    audit_run_id = make_run_id({
        "phase": "phase5", "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": implementation_sha, "partition_mode": partition_mode,
        "dataset": "cifar10", "cases": list(cases), "seeds": list(seeds),
        "model_config_sha256": model_config_sha,
    })
    prompt_sha = hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    audit_results, audit_specs, private_episodes = [], {}, []
    terminal_failures = []
    total_jobs = 0

    for seed in seeds:
        for case_id in cases:
            cfg, data, plan = prepare_episode(
                case_id, seed, False, False, DEFAULT_MANIFEST,
                out_dir / "private" / "private_plans" / f"{case_id}_s{seed}.json",
                True, partition_mode,
            )
            event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
            deterministic = _phase5_roster(
                seed, event_round, plan["hidden_event"]["canonical_bundle"]
            )
            policy = {
                **manifest["protocol"]["phase6"]["agent_policy"],
                "decision_cadence": int(cfg.decide_every),
                "max_calls": (int(cfg.rounds) + int(cfg.decide_every) - 1) //
                             int(cfg.decide_every),
            }
            llm_entries = [
                _configured_llm_agent(model, policy, implementation_sha)
                for model in model_specs
            ]
            agents = deterministic + [agent for agent, _ in llm_entries]
            specs = {
                agent.name: {
                    "name": agent.name, "agent_version": implementation_sha,
                    "model": "deterministic",
                    "policy": {"decision_cadence": int(cfg.decide_every)},
                }
                for agent in deterministic
            }
            specs.update({agent.name: spec for agent, spec in llm_entries})
            histories, fingerprints = {}, {}

            for agent in agents:
                spec = specs[agent.name]
                checkpoint = (
                    out_dir / "private" / "checkpoints" / plan["episode_id"] /
                    f"{agent.name}.json"
                )
                fingerprint = run_fingerprint(
                    manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
                    data=data, agent=agent, synthetic=False, smoke=False,
                    prompt_sha=prompt_sha,
                )
                if agent in deterministic:
                    payload = (json.loads(checkpoint.read_text(encoding="utf-8"))
                               if checkpoint.exists() else None)
                    if payload is not None:
                        expected = {
                            "episode_id": plan["episode_id"], "agent": agent.name,
                            "run_fingerprint_sha256": fingerprint["sha256"],
                            "agent_spec": spec,
                        }
                        if any(payload.get(key) != value for key, value in expected.items()):
                            raise ValueError(f"deterministic checkpoint mismatch: {checkpoint}")
                        history = payload["history"]
                    else:
                        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
                        if terminal is None:
                            raise FileNotFoundError(checkpoint)
                        integrity = dict(terminal["integrity"])
                        integrity.update({
                            "run_id": audit_run_id,
                            "agent_config_id": agent_config_id(spec),
                        })
                        terminal_failures.append(integrity)
                        audit_results.append({
                            "integrity": integrity, "events": [],
                            "decisions": [], "artifacts": {},
                        })
                        audit_specs[agent_config_id(spec)] = spec
                        fingerprints[agent.name] = fingerprint["sha256"]
                        total_jobs += 1
                        continue
                else:
                    history = _run_or_record_failure(
                        checkpoint, agent, cfg, data, plan, fingerprint, spec,
                        "B3_API", audit_run_id,
                    )
                    if history is None:
                        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
                        integrity = dict(terminal["integrity"])
                        integrity.update({
                            "run_id": audit_run_id,
                            "agent_config_id": agent_config_id(spec),
                        })
                        terminal_failures.append(integrity)
                        audit_results.append({
                            "integrity": integrity, "events": [],
                            "decisions": [], "artifacts": {},
                        })
                        audit_specs[agent_config_id(spec)] = spec
                        fingerprints[agent.name] = fingerprint["sha256"]
                        total_jobs += 1
                        continue

                histories[agent.name] = history
                fingerprints[agent.name] = fingerprint["sha256"]
                audit_specs[agent_config_id(spec)] = spec
                audit_results.append(audit_episode(
                    audit_run_id, agent.name, spec, plan, history, manifest,
                ))
                total_jobs += 1

            hidden_logs = [history["hidden_event_log"] for history in histories.values()]
            if any(log != hidden_logs[0] for log in hidden_logs[1:]):
                raise ValueError("the same episode changed hidden targets across agents")
            causal = histories.get("causal_oracle")
            if causal is None:
                raise RuntimeError(f"causal_oracle failed for {case_id}/seed={seed}")
            private_episodes.append({
                "case_id": case_id, "seed": seed, "episode_plan": plan,
                "run_fingerprints": fingerprints,
                "mechanism_evidence": _validate_mechanism(
                    plan, causal, event_round,
                    plan["hidden_event"]["canonical_bundle"],
                ),
            })

    metadata = {
        "schema_version": manifest["schema_version"],
        "evaluation_standard": "agent_decision_audit",
        "run_id": audit_run_id, "implementation_sha256": implementation_sha,
        "manifest_sha256": manifest_sha256(manifest),
        "model_config_sha256": model_config_sha,
        "models": [model["model"] for model in model_specs],
        "dataset": "cifar10", "partition_mode": partition_mode,
        "smoke": False, "seeds": list(seeds),
        "episode_count": len(private_episodes), "executed_jobs": total_jobs,
        "terminal_failure_count": len(terminal_failures),
        "runner_git_sha": git["git_sha"], "git_dirty": git["git_dirty"],
        "git_status_sha256": git["git_status_sha256"],
    }
    summaries = _write_agent_decision_audit(
        out_dir, audit_results, audit_specs, metadata
    )
    retryable = [
        result["integrity"] for result in audit_results
        if result["integrity"]["run_status"] != "VALID" and
        result["integrity"].get("rerun_required", True)
    ]
    if retryable:
        failed = ", ".join(
            f"{row['agent_id']}/{row['episode_id']}:{row['failure_type']}"
            for row in retryable
        )
        raise RetryableRunError(f"formal Agent decision audit requires reruns: {failed}")
    _write_b3_json(out_dir / "private" / "phase5_audit.json", {
        "schema_version": manifest["schema_version"],
        "metadata": metadata, "episodes": private_episodes,
        "terminal_failures": terminal_failures,
    })
    _write_b3_json(out_dir / "run_metadata.json", metadata)
    print(
        f"B3_API_ONLY_OK episodes={len(private_episodes)} jobs={total_jobs} "
        f"terminal_failures={len(terminal_failures)} summaries={len(summaries)}",
        flush=True,
    )


def main(*, fixed_stage: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_stage is None:
        parser.add_argument(
            "--stage", choices=("check", "deterministic", "api", "all"), default="all"
        )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/b3_composite_v2/formal_eval_low_v1"),
    )
    parser.add_argument("--partition-mode", choices=("iid", "noniid"), default="noniid")
    parser.add_argument("--retry-passes", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stage = fixed_stage or args.stage
    if args.retry_passes < 1:
        raise ValueError("--retry-passes must be at least 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    if stage in {"api", "all"} and (args.num_shards, args.shard_index) != (1, 0):
        raise ValueError("sharding is supported only by the deterministic stage")

    manifest = load_manifest(DEFAULT_MANIFEST)
    require_formal_manifest(manifest)
    cases = list(manifest["protocol"]["formal_case_ids"])
    seeds = list(manifest["protocol"]["evaluation_seeds"])
    if len(cases) != 10 or seeds != [0, 44, 56]:
        raise ValueError("the frozen formal queue is not 10 cases x seeds 0/44/56")
    shard_order = [
        "hetero_abrupt_dropout", "virtual_hetero",
        "real_hetero", "staggered",
        "real_dropout", "real_fault",
        "virtual_fault", "virtual_dropout",
        "label_dropout", "fault_dropout",
    ]
    if set(shard_order) != set(cases):
        raise ValueError("non-API shard order drifted from the frozen case roster")
    selected_cases = shard_order[args.shard_index::args.num_shards]
    if args.check or stage == "check":
        _self_check_terminal_failure()
        print(
            f"B3_FORMAL_QUEUE_CHECK_OK stage={stage} shard={args.shard_index}/"
            f"{args.num_shards} cases={','.join(selected_cases)} seeds=0,44,56 "
            f"deterministic_jobs={len(selected_cases) * len(seeds) * 12} "
            "api_jobs=60 models=qwen3.8-flash,deepseek-v4-pro-0813"
        )
        return 0

    out_dir = partition_artifact_path(args.out_dir, manifest, args.partition_mode)
    require_versioned_output(out_dir, manifest, args.partition_mode)
    if stage in {"deterministic", "all"}:
        deterministic_stage(manifest, out_dir, selected_cases, seeds, args.partition_mode)
    if stage in {"api", "all"}:
        api_stage(manifest, out_dir, cases, seeds, args.partition_mode, args.retry_passes)
    print(f"B3_FORMAL_QUEUE_OK stage={stage} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
