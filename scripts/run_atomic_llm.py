"""Run one frozen-protocol single-cause LLM episode for the ICASSP atomic table."""

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

from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256  # noqa: E402


ATOMIC_CASES = {
    "real": ("real_drift", "drift_adapt"),
    "virtual": ("virtual_drift", "moment_align_adapt"),
    "label": ("label_prior_drift", "label_prior_adapt"),
    "fault": ("fault", "robust"),
    "dropout": ("dropout", "dropout_handle"),
    "hetero": ("partial_participation_hetero", "select_clients"),
}
FROZEN_MANIFEST_SHA = "af5afd16150f4378edeace0ad00151904c7fb8c6d15e8bcebb9ee9a8e3eb540d"


def action_set(row):
    return set(row.get("action_bundle", "no_op").split("|")) - {"no_op"}


def agent_name(model, effort):
    known = {
        ("qwen3.8-flash", "low"): "qwen3p8_flash_low",
        ("glm-5.2", "low"): "glm_5p2_low",
        ("deepseek-v4-flash-0731", "low"): "deepseek_v4_flash_0731_low",
    }
    return known.get((model, effort), model.replace("-", "_").replace(".", "p") + f"_{effort}")


def atomic_manifest(path: Path):
    base = load_manifest(DEFAULT_MANIFEST)
    derived = copy.deepcopy(base)
    derived["atomic_extension"] = {
        "base_manifest_sha256": manifest_sha256(base),
        "cases": list(ATOMIC_CASES),
    }
    derived["formal_cases"] += [
        {
            "case_id": f"atomic_{name}",
            "causes": [cause],
            "canonical_bundle": [action],
            "feasible_bundles": [[action]],
        }
        for name, (cause, action) in ATOMIC_CASES.items()
    ]
    derived["integrity"]["manifest_sha256"] = manifest_sha256(derived)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != derived:
            raise ValueError(f"atomic manifest mismatch: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    return path


def journal_calls(checkpoint):
    path = checkpoint.with_suffix(".llm_journal.jsonl")
    return ([] if not path.exists() else
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])


def run(args):
    from b3_composite import _configured_llm_agent, _write_json, prepare_episode
    from b3_fingerprint import run_fingerprint, source_sha256
    from llm_backend import B3_SYSTEM_PROMPT
    from scripts.run_b3_formal_eval import _run_or_record_failure, _terminal_failure

    base_sha = manifest_sha256(load_manifest(DEFAULT_MANIFEST))
    if base_sha != FROZEN_MANIFEST_SHA:
        raise RuntimeError(
            f"formal atomic run requires frozen manifest {FROZEN_MANIFEST_SHA}; got {base_sha}"
        )
    case_id = f"atomic_{args.case}"
    episode_dir = args.out_dir / f"{case_id}_s{args.seed}"
    manifest_path = atomic_manifest(args.out_dir / "private" / "atomic_manifest.json")
    plan_path = episode_dir / "private" / "private_plans" / f"{case_id}_s{args.seed}.json"
    cfg, data, plan = prepare_episode(
        case_id, args.seed, False, False, manifest_path=manifest_path,
        plan_path=plan_path, resume=plan_path.exists(),
    )
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("missing DASHSCOPE_API_KEY")
    model = {
        "name": agent_name(args.model, args.reasoning_effort), "kind": "closed",
        "model": args.model, "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY", "requires_api_key": True,
        "temperature": args.temperature, "reasoning_effort": args.reasoning_effort,
        "input_cost_per_million": None, "output_cost_per_million": None,
    }
    manifest = load_manifest(manifest_path)
    policy = {**manifest["protocol"]["phase6"]["agent_policy"],
              "decision_cadence": int(cfg.decide_every), "max_calls": int(cfg.rounds)}
    agent, spec = _configured_llm_agent(
        model, policy, source_sha256(), ablation_group=args.ablation_group
    )
    checkpoint = episode_dir / "private" / "checkpoints" / plan["episode_id"] / f"{agent.name}.json"
    fingerprint = run_fingerprint(
        manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg, data=data,
        agent=agent, synthetic=False, smoke=False,
        prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    )
    history = _run_or_record_failure(
        checkpoint, agent, cfg, data, plan, fingerprint, spec,
        "ATOMIC_LLM", "icassp-atomic-v1",
    )
    calls = journal_calls(checkpoint)
    expected = set(plan["hidden_event"]["canonical_bundle"])
    result = {
        "case": args.case, "case_id": case_id, "seed": args.seed,
        "model": args.model, "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature, "expected": sorted(expected),
        "rounds": int(cfg.rounds), "api_calls": len(calls),
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in calls),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in calls),
    }
    if history is None:
        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
        result.update({"status": "terminal_failure", "terminal_failure":
                       None if terminal is None else terminal["integrity"]})
    else:
        event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
        window = history["telemetry"][event_round:event_round + 10]
        post = history["telemetry"][event_round + 10:]
        exact = [int(row["round"]) for row in window if action_set(row) == expected]
        result.update({
            "status": "completed", "first_exact_round": exact[0] if exact else None,
            "success_at_10": bool(exact) and not any(action_set(row) - expected for row in window),
            "exact_decisions_at_10": len(exact),
            "correct_rounds_50_79": sum(action_set(row) == expected for row in post),
            "round_79_actions": sorted(action_set(history["telemetry"][-1])),
            "interface_error_rounds": [int(row["round"]) for row in history["telemetry"]
                if row.get("agent_interface_error") or row.get("llm_api_error") or
                row.get("llm_parse_error") or row.get("llm_budget_error")],
        })
    _write_json(episode_dir / "summary.json", result)
    print(f"ATOMIC_LLM_OK case={args.case} seed={args.seed} status={result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--case", choices=tuple(ATOMIC_CASES), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 44, 56), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "atomic_llm")
    parser.add_argument("--ablation-group", choices=("G1", "G2", "G3", "G4", "G5"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        from agent_decision_audit import compile_events
        from b3_episode_plan import build_episode_plan, resolve_episode_clients
        import tempfile

        assert len(ATOMIC_CASES) == 6
        with tempfile.TemporaryDirectory() as tmp:
            manifest = atomic_manifest(Path(tmp) / "atomic_manifest.json")
            plan = build_episode_plan("atomic_hetero", 0, manifest_path=manifest)
            assert plan["hidden_event"]["event_schedule"][0]["round"] == 0
            count = plan["hidden_event"]["affected_clients"][
                "partial_participation_hetero"
            ]["planned_count"]
            plan = resolve_episode_clients(plan, planned_clients=list(range(count)))
            events = compile_events(
                plan, load_manifest(manifest),
                [{"round": round_idx, "decision_due": True} for round_idx in range(80)],
            )
            window = int(load_manifest(manifest)["protocol"]["response_window_length"])
            assert len(events) == 1 and events[0]["emit_deadline"] == window - 1
        print("ATOMIC_LLM_CHECK_OK cases=real,virtual,label,fault,dropout,hetero seeds=0,44,56")
        return
    run(args)


if __name__ == "__main__":
    main()
