"""Run the frozen B1 suite, robustness cases, and optional real-model evaluation."""
import argparse
import csv
import json
import os
import subprocess
import sys

from agents import LLMAgent
from observation_contract import serialize_llm_observation
from telemetry import collect


ROOT = os.path.dirname(os.path.abspath(__file__))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


def leakage_audit():
    manifest_path = os.path.join(ROOT, "benchmark_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    required = {"id", "correct_action", "primary_metrics", "guardrails"}
    if len(manifest["cases"]) != 7:
        raise ValueError("benchmark case count mismatch")
    if not all(required <= set(case) for case in manifest["cases"]):
        raise ValueError("benchmark case schema mismatch")
    if len({case["id"] for case in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("duplicate benchmark case id")

    obs = collect(7, 0.5, [0.6, 0.55, 0.5, 0.5], [1.0, 1.2], [1.0, 1.1],
                  0.01, 1.0, False, 80, input_shift=0.2, label_shift=0.1)
    captured = []
    LLMAgent(query_fn=lambda text: captured.append(text) or '{"actions":["no_op"]}').decide(obs)
    if captured != [serialize_llm_observation(obs)]:
        raise ValueError("LLM observation rendering mismatch")
    projected = json.loads(captured[0])
    if ("client_utilities" in projected or "input_shift" in projected or
            "label_shift" in projected):
        raise ValueError("LLM observation was not compacted or neutrally named")
    if not {"input_moment_delta", "aggregate_label_hist_tv", "utility_mean",
            "client_model_score_drop_p50", "client_model_score_drop_p90",
            "client_update_direction_dispersion", "online_roster_overlap",
            "client_score_exceedance_fraction", "client_score_exceedance_overlap",
            "client_update_group_separation"} <= set(projected):
        raise ValueError("LLM observation projection is missing required summaries")
    lowered = captured[0].lower()
    forbidden = ("scenario=", "drift_type", "drift_round", "fault", "dropout",
                 "staggered", "concept_of", "drift_clients", "feddrift",
                 "select_seed", "oort_")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"public observation leakage: {lowered}")
    print("LEAKAGE_AUDIT_OK")


def model_commands(path, common, out_dir, validate_keys=True):
    with open(path, encoding="utf-8") as f:
        models = json.load(f).get("models", [])
    commands = []
    for model in models:
        env_name = model.get("api_key_env", "OPENAI_API_KEY")
        if validate_keys and model.get("requires_api_key", True) and not os.environ.get(env_name):
            raise SystemExit(f"missing {env_name} for model {model['name']}")
        if (validate_keys and model.get("requires_api_key", True)
                and not model.get("allow_zero_cost", False)
                and not (model.get("input_cost_per_million") or model.get("output_cost_per_million"))):
            raise SystemExit(
                f"missing token prices for {model['name']}; set prices or allow_zero_cost=true"
            )
        command = common + [
            "--b1_full", "--agent_names", "noop", "oracle", "llm",
            "--decide_every", "10", "--llm_model", model["model"],
            "--llm_api_key_env", env_name,
            "--llm_temperature", str(model.get("temperature", 0.0)),
            "--llm_input_cost_per_million", str(model.get("input_cost_per_million", 0.0)),
            "--llm_output_cost_per_million", str(model.get("output_cost_per_million", 0.0)),
            "--out_dir", os.path.join(out_dir, "agents", model["name"]),
        ]
        if model.get("base_url"):
            command += ["--llm_base_url", model["base_url"]]
        commands.append(command)
    return commands


def aggregate_agent_results(out_dir, models_path):
    sources = [("rule_diagnose", "rule", "diagnose",
                os.path.join(out_dir, "agents", "rule_diagnose"))]
    if models_path:
        with open(models_path, encoding="utf-8") as f:
            for model in json.load(f).get("models", []):
                sources.append((model["name"], model.get("kind", "model"), "llm",
                                os.path.join(out_dir, "agents", model["name"])))
    leaderboard, confusion = [], []
    for name, kind, agent, directory in sources:
        metrics_path = os.path.join(directory, "metrics_all_multiseed.csv")
        confusion_path = os.path.join(directory, "action_confusion.csv")
        if os.path.exists(metrics_path):
            metric_rows = read_csv(metrics_path)
            by_case_agent = {(row["case"], row["agent"]): row for row in metric_rows}
            for row in metric_rows:
                if row["agent"] != agent:
                    continue
                primary = row["primary_metric"]
                noop = by_case_agent.get((row["case"], "noop"), {})
                oracle = by_case_agent.get((row["case"], "oracle"), {})
                try:
                    score = float(row[f"{primary}_mean"])
                    noop_score = float(noop[f"{primary}_mean"])
                    oracle_score = float(oracle[f"{primary}_mean"])
                    denominator = oracle_score - noop_score
                    regret_ratio = ((oracle_score - score) / denominator
                                    if denominator >= 0.01 else "n/a")
                except (KeyError, TypeError, ValueError):
                    regret_ratio = "n/a"
                leaderboard.append({
                    "model": name, "kind": kind, "case": row["case"],
                    "primary_metric": primary, "primary_score": row.get(f"{primary}_mean", ""),
                    "delta_vs_noop": row.get("primary_delta_vs_noop_mean", ""),
                    "win_rate_vs_noop": row.get("primary_win_rate_vs_noop", ""),
                    "misdiagnosis_penalty": row.get("misdiagnosis_penalty_mean", ""),
                    "active_action_count": row.get("active_action_count_mean", ""),
                    "wrong_action_count": row.get("wrong_action_count_mean", ""),
                    "regret_ratio": round(regret_ratio, 4) if isinstance(regret_ratio, float) else regret_ratio,
                    "switch_regret": row.get("switch_regret_mean", ""),
                    "over_intervention": row.get("over_intervention_mean", ""),
                    "llm_calls": row.get("llm_calls_mean", ""),
                    "llm_total_tokens": row.get("llm_total_tokens_mean", ""),
                    "llm_cost_usd": row.get("llm_cost_usd_mean", ""),
                    "llm_latency_ms_mean": row.get("llm_latency_ms_mean_mean", ""),
                    "llm_latency_ms_p95": row.get("llm_latency_ms_p95_mean", ""),
                })
        if os.path.exists(confusion_path):
            for row in read_csv(confusion_path):
                if row["agent"] == agent:
                    confusion.append({"model": name, "kind": kind, **row})
    paths = [write_csv(os.path.join(out_dir, "agent_leaderboard.csv"), leaderboard),
             write_csv(os.path.join(out_dir, "agent_action_confusion.csv"), confusion)]
    return [path for path in paths if path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 44, 56])
    parser.add_argument("--out_dir", default="results/benchmark_cifar10")
    parser.add_argument("--models", help="JSON model list; use benchmark_models.example.json as a template")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip_b1", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_agents", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    leakage_audit()
    common = [sys.executable, os.path.join(ROOT, "multi_seed.py"),
              "--seeds", *map(str, args.seeds), "--formal_evidence"]
    if args.synthetic:
        common.append("--synthetic")
    if args.smoke:
        common.append("--smoke")

    commands = []
    if not args.skip_b1:
        commands.append(common + ["--b1_full", "--out_dir", os.path.join(args.out_dir, "b1_final")])
    if not args.skip_robustness:
        commands.append(common + ["--scenarios", "drift", "--drift_types", "real", "virtual", "label",
                                  "--drift_mode", "gradual", "--out_dir",
                                  os.path.join(args.out_dir, "robustness", "gradual")])
        for rate in (0.3, 0.5, 0.7):
            commands.append(common + ["--scenarios", "dropout", "--dropout_mode", "random",
                                      "--dropout_rate", str(rate), "--out_dir",
                                      os.path.join(args.out_dir, "robustness", f"dropout_random_{int(rate * 100)}")])
    if not args.skip_agents:
        if not args.models:
            raise SystemExit("--models is required unless --skip_agents is set")
        commands.append(common + ["--b1_full", "--agent_names", "noop", "random",
                                  "diagnose", "oracle", "--decide_every", "10",
                                  "--out_dir", os.path.join(args.out_dir, "agents", "rule_diagnose")])
        commands.extend(model_commands(args.models, common, args.out_dir,
                                       validate_keys=not args.dry_run))

    os.makedirs(args.out_dir, exist_ok=True)
    plan_path = os.path.join(args.out_dir, "suite_commands.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({"commands": commands}, f, ensure_ascii=False, indent=2)
    print(f"saved {plan_path}")
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        for path in aggregate_agent_results(args.out_dir, args.models):
            print(f"saved {path}")


if __name__ == "__main__":
    main()
