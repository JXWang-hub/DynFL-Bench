"""Run selected API models on one real-CIFAR B3 episode and estimate 30-job cost."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from b3_composite import _configured_llm_agent, _write_json, prepare_episode  # noqa: E402
from b3_fingerprint import run_fingerprint, source_sha256  # noqa: E402
from b3_manifest import load_manifest, manifest_sha256  # noqa: E402
from llm_backend import B3_SYSTEM_PROMPT, make_openai_query_fn  # noqa: E402
from scripts.run_b3_formal_eval import (  # noqa: E402
    _run_or_record_failure, _terminal_failure,
)


MODELS = {
    "qwen3p8_flash_low": {
        "model": "qwen3.8-flash", "reasoning_effort": "low",
        "prices": {"standard": (0.8, 0.1, 2.7)},
    },
    "minimax_m2p5": {
        "model": "MiniMax-M2.5", "reasoning_effort": None,
        "prices": {"standard": (2.1, 0.42, 8.4)},
    },
    "glm_5p2_low": {
        "model": "glm-5.2", "reasoning_effort": "low",
        "prices": {"standard": (8.0, 2.0, 28.0)},
    },
    "deepseek_v4_flash_0731_low": {
        "model": "deepseek-v4-flash-0731", "reasoning_effort": "low",
        "prices": {
            "busy": (3.0, 0.3, 9.0),
            "idle": (1.5, 0.15, 4.5),
        },
    },
    "qwen3p8_flash_xhigh": {
        "model": "qwen3.8-flash", "reasoning_effort": "xhigh",
        "prices": {"standard": (0.8, 0.1, 2.7)},
    },
    "qwen3p8_max_0902_low": {
        "model": "qwen3.8-max-0902", "reasoning_effort": "low",
        "prices": {"standard": (12.0, 1.5, 36.0)},
    },
}


def action_set(row):
    return set(row.get("action_bundle", "no_op").split("|")) - {"no_op"}


def journal_calls(checkpoint):
    path = checkpoint.with_suffix(".llm_journal.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def costs(prompt_tokens, completion_tokens, profiles):
    result = {}
    for name, (input_price, cached_price, output_price) in profiles.items():
        result[name] = {
            "one_job_all_uncached_cny": round(
                (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000,
                6,
            ),
            "one_job_all_cached_cny": round(
                (prompt_tokens * cached_price + completion_tokens * output_price) / 1_000_000,
                6,
            ),
            "thirty_jobs_all_uncached_cny": round(
                30 * (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000,
                4,
            ),
            "thirty_jobs_all_cached_cny": round(
                30 * (prompt_tokens * cached_price + completion_tokens * output_price) / 1_000_000,
                4,
            ),
        }
    return result


def model_config(args, model_name):
    configured = MODELS[model_name]
    return {
        "name": model_name,
        "kind": "closed",
        "model": configured["model"],
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "requires_api_key": True,
        "temperature": args.temperature,
        "reasoning_effort": configured["reasoning_effort"],
        "input_cost_per_million": None,
        "output_cost_per_million": None,
    }


def preflight(args):
    results = []
    for model_name in args.models:
        model = model_config(args, model_name)
        query = make_openai_query_fn(
            model=model["model"], base_url=model["base_url"],
            api_key_env=model["api_key_env"], temperature=model["temperature"],
            system_prompt=B3_SYSTEM_PROMPT, timeout_seconds=120,
            retry_delays_seconds=(2,), reasoning_effort=model["reasoning_effort"],
        )
        output = query('{"round":0,"global_acc":0.5}')
        row = {
            "name": model_name, "model": model["model"],
            "reasoning_effort": model["reasoning_effort"],
            "error": query.last_error, "attempts": query.last_attempts,
            "usage": query.last_usage, "latency_ms": round(query.last_latency_ms, 3),
            "output": output,
        }
        results.append(row)
        print(
            f"MODEL_PROBE_PREFLIGHT model={model_name} error={query.last_error or 'none'} "
            f"attempts={query.last_attempts}", flush=True,
        )
        if query.last_error:
            raise RuntimeError(f"preflight failed for {model_name}: {query.last_error}")
    _write_json(args.out_dir / "preflight.json", {"results": results})


def run_model(args, model_name, manifest, plan_path):
    configured = MODELS[model_name]
    model = model_config(args, model_name)
    if not os.environ.get(model["api_key_env"]):
        raise RuntimeError(f"missing {model['api_key_env']}")

    cfg, data, plan = prepare_episode(
        args.case, args.seed, False, False, plan_path=plan_path,
        resume=plan_path.exists(),
    )
    policy = {
        **manifest["protocol"]["phase6"]["agent_policy"],
        "decision_cadence": int(cfg.decide_every),
        "max_calls": int(cfg.rounds),
    }
    agent, spec = _configured_llm_agent(model, policy, source_sha256())
    checkpoint = (
        args.out_dir / "private" / "checkpoints" / plan["episode_id"] /
        f"{agent.name}.json"
    )
    fingerprint = run_fingerprint(
        manifest_sha=manifest_sha256(manifest), plan=plan, cfg=cfg,
        data=data, agent=agent, synthetic=False, smoke=False,
        prompt_sha=hashlib.sha256(B3_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    )
    history = _run_or_record_failure(
        checkpoint, agent, cfg, data, plan, fingerprint, spec,
        "MODEL_PROBE", "four-model-real-dropout-s0-v1",
    )
    calls = journal_calls(checkpoint)
    prompt_tokens = sum(int(row["usage"]["prompt_tokens"]) for row in calls)
    completion_tokens = sum(int(row["usage"]["completion_tokens"]) for row in calls)
    base = {
        "name": model_name,
        "model": model["model"],
        "reasoning_effort": model["reasoning_effort"],
        "temperature": model["temperature"],
        "case": args.case,
        "seed": args.seed,
        "rounds": int(cfg.rounds),
        "api_calls": len(calls),
        "api_attempts": sum(int(row.get("attempts", 1)) for row in calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mean_latency_ms": round(
            sum(float(row["latency_ms"]) for row in calls) / max(1, len(calls)), 3
        ),
        "cost_estimates": costs(prompt_tokens, completion_tokens, configured["prices"]),
    }
    if history is None:
        terminal = _terminal_failure(checkpoint, fingerprint["sha256"])
        return {**base, "status": "terminal_failure", "terminal_failure": terminal}

    event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
    expected = set(plan["hidden_event"]["canonical_bundle"])
    window = history["telemetry"][event_round:event_round + 10]
    post = history["telemetry"][event_round + 10:]
    exact = [int(row["round"]) for row in window if action_set(row) == expected]
    extra = [int(row["round"]) for row in window if action_set(row) - expected]
    return {
        **base,
        "status": "completed",
        "expected": sorted(expected),
        "first_exact_round": exact[0] if exact else None,
        "success_at_10": bool(exact) and not extra,
        "exact_decisions_at_10": len(exact),
        "extra_action_rounds_at_10": extra,
        "correct_rounds_50_79": sum(action_set(row) == expected for row in post),
        "round_79_actions": sorted(action_set(history["telemetry"][-1])),
        "interface_error_rounds": [
            int(row["round"]) for row in history["telemetry"]
            if row.get("agent_interface_error") or row.get("llm_api_error") or
            row.get("llm_parse_error") or row.get("llm_budget_error")
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--case", default="real_dropout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "results" / "four_model_probe_real_dropout_s0",
    )
    args = parser.parse_args()
    if args.check:
        print(
            f"MODEL_PROBE_CHECK_OK models={','.join(args.models)} case={args.case} "
            f"seed={args.seed} temperature={args.temperature} rounds=80"
        )
        return
    if args.preflight:
        preflight(args)
        print(f"MODEL_PROBE_PREFLIGHT_OK models={len(args.models)}", flush=True)
        return

    manifest = load_manifest()
    plan_path = args.out_dir / "private" / "private_plans" / f"{args.case}_s{args.seed}.json"
    results = []
    for model_name in args.models:
        result = run_model(args, model_name, manifest, plan_path)
        results.append(result)
        _write_json(args.out_dir / "summary.json", {
            "status": "running", "completed": len(results),
            "total": len(args.models), "results": results,
        })
        print(
            f"MODEL_PROBE_PROGRESS completed={len(results)}/{len(args.models)} "
            f"model={model_name} status={result['status']} calls={result['api_calls']}",
            flush=True,
        )
    _write_json(args.out_dir / "summary.json", {
        "status": "completed", "completed": len(results),
        "total": len(args.models), "results": results,
    })
    print(f"MODEL_PROBE_OK completed={len(results)}/{len(args.models)}", flush=True)


if __name__ == "__main__":
    main()
