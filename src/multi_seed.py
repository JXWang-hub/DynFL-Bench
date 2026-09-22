"""Multi-seed runner for B1 acceptance.

Examples:
  conda run -n dynfl python multi_seed.py
  conda run -n dynfl python multi_seed.py --scenarios drift --drift_types virtual label
  conda run -n dynfl python multi_seed.py --b1_full --seeds 0 44 56
  conda run -n dynfl python multi_seed.py --synthetic --smoke --b1_full

The script reuses run.build_data / run.set_seed and the same agent roster as
run.py, so the per-seed numbers match a normal `run.py --seed S` exactly.
It writes per-case CSVs, recovery curves, one combined CSV, and heatmap PNGs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

from agents import CompositeDiagnoseAgent, make_agents
from config import (Config, DATASETS, DRIFT_FRACTION, PARTITION_MODES, partition_output_dir)
from engine import Engine
from run import build_data, set_seed
from scoring import summarize

SEEDS = [0, 44, 56]
DEFAULT_SCENARIOS = ["drift", "fault", "dropout"]
SCENARIOS = ["drift", "fault", "dropout", "hetero", "staggered"]
DRIFT_TYPES = ["real", "virtual", "label"]
B1_FULL_CASES = [
    ("drift", "real"),
    ("drift", "virtual"),
    ("drift", "label"),
    ("fault", None),
    ("dropout", None),
    ("hetero", None),
    ("staggered", None),
]
METRICS = [
    "final_acc",
    "post_drift_mean_acc",
    "post_event_worst_acc",
    "transition_mean_acc",
    "post_full_drift_mean_acc",
    "recovery_rounds",
    "misdiagnosis_penalty",
    "active_action_count",
    "wrong_action_count",
    "over_intervention",
    "switch_round",
    "trigger_target_round",
    "switch_regret",
    "switch_update_norm_ratio",
    "selected_action_rate",
    "feddrift_cluster_purity",
    "feddrift_num_concepts",
    "feddrift_first_trigger_round",
    "feddrift_assignment_changed_clients",
    "llm_calls",
    "llm_total_tokens",
    "llm_cost_usd",
    "llm_latency_ms_mean",
    "llm_latency_ms_p95",
]
PERFORMANCE_BOUND_AGENTS = {"staggered": {"spawn_concept"}}


def case_name(scenario, drift_type=None, drift_mode="sudden", dropout_mode=None, dropout_rate=None):
    if scenario == "drift" and drift_type:
        mode = "" if drift_mode == "sudden" else f"{drift_mode}_"
        return f"drift_{mode}{drift_type}"
    if scenario == "dropout" and dropout_mode and dropout_mode != "correlated":
        rate = f"_{int(round(100 * dropout_rate))}" if dropout_rate is not None else ""
        return f"dropout_{dropout_mode}{rate}"
    return scenario


def primary_metric_for(case):
    return ("post_event_worst_acc"
            if case.startswith("drift_") and case.endswith(("virtual", "label"))
            else "post_drift_mean_acc")


def target_agent_for(case):
    if case.startswith("drift_"):
        if "_recurrent_" in case:
            return None
        return "label_prior_adapt" if case.endswith("label") else "diagnose" if case.endswith("virtual") else "adapt_lr"
    if case.startswith("dropout"):
        return "handle_dropout"
    return {"fault": "reject_robust", "hetero": "select_clients",
            "staggered": "spawn_concept_auto"}.get(case)


def selected_cases(args):
    if args.b1_full:
        return B1_FULL_CASES

    cases = []
    for scenario in args.scenarios:
        if scenario == "drift":
            for drift_type in args.drift_types or ["real"]:
                cases.append((scenario, drift_type))
        else:
            cases.append((scenario, None))
    return cases


def make_cfg(
    scenario,
    seed,
    synthetic,
    smoke,
    dataset="cifar10",
    formal_evidence=False,
    proxy_evidence=False,
    drift_type=None,
    virtual_shift=None,
    dropout_mode=None,
    dropout_rate=None,
    fedlc_tau=None,
    feddaa_T=None,
    label_imbalance_factor=None,
    label_min_per_class=None,
    label_prior_strength=None,
    out_dir="results",
    drift_mode=None,
    decide_every=None,
    partition_mode="noniid",
):
    cfg = Config()
    cfg.scenario = scenario
    cfg.seed = seed
    cfg.out_dir = out_dir
    cfg.partition_mode = partition_mode
    cfg.formal_evidence = formal_evidence
    cfg.proxy_evidence = proxy_evidence
    cfg.drift_fraction = DRIFT_FRACTION[scenario]
    cfg.dataset = dataset
    if scenario == "drift" and drift_type is not None:
        cfg.drift_type = drift_type
    if virtual_shift is not None:
        cfg.virtual_shift = virtual_shift
    if drift_mode is not None:
        cfg.drift_mode = drift_mode
    if dropout_mode is not None:
        cfg.dropout_mode = dropout_mode
    if dropout_rate is not None:
        cfg.dropout_rate = dropout_rate
    if fedlc_tau is not None:
        cfg.fedlc_tau = fedlc_tau
    if feddaa_T is not None:
        cfg.feddaa_T = feddaa_T
    if label_imbalance_factor is not None:
        cfg.label_imbalance_factor = label_imbalance_factor
    if label_min_per_class is not None:
        cfg.label_min_per_class = label_min_per_class
    if label_prior_strength is not None:
        cfg.label_prior_strength = label_prior_strength
    if decide_every is not None:
        cfg.decide_every = decide_every
    if scenario == "hetero":
        cfg.local_epochs = cfg.hetero_local_epochs
        cfg.participation = cfg.hetero_participation
    if synthetic:
        cfg.dataset = "synthetic"
    if smoke:
        cfg.smoke()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def run_one(
    scenario,
    seed,
    synthetic,
    smoke,
    dataset="cifar10",
    formal_evidence=False,
    proxy_evidence=False,
    drift_type=None,
    virtual_shift=None,
    dropout_mode=None,
    dropout_rate=None,
    fedlc_tau=None,
    feddaa_T=None,
    label_imbalance_factor=None,
    label_min_per_class=None,
    label_prior_strength=None,
    out_dir="results",
    drift_mode=None,
    decide_every=None,
    agent_names=None,
    llm_settings=None,
    partition_mode="noniid",
):
    cfg = make_cfg(
        scenario=scenario, seed=seed, synthetic=synthetic, smoke=smoke,
        dataset=dataset,
        formal_evidence=formal_evidence, proxy_evidence=proxy_evidence,
        drift_type=drift_type, virtual_shift=virtual_shift,
        dropout_mode=dropout_mode, dropout_rate=dropout_rate,
        fedlc_tau=fedlc_tau, feddaa_T=feddaa_T,
        label_imbalance_factor=label_imbalance_factor,
        label_min_per_class=label_min_per_class, label_prior_strength=label_prior_strength,
        out_dir=out_dir, drift_mode=drift_mode, decide_every=decide_every,
        partition_mode=partition_mode,
    )
    event_round = 0 if scenario == "hetero" else cfg.drift_round
    set_seed(seed)
    data = build_data(cfg)
    eng = Engine(cfg, data)
    llm_query_fn = None
    if llm_settings:
        from llm_backend import make_openai_query_fn
        llm_query_fn = make_openai_query_fn(**llm_settings)
    agents = make_agents(scenario, seed, event_round, llm_query_fn=llm_query_fn,
                         drift_type=cfg.drift_type)
    if agent_names and "composite_rule" in agent_names:
        agents.append(CompositeDiagnoseAgent())
    if agent_names:
        agents = [agent for agent in agents if agent.name in agent_names]
    out = {}
    curves = {}
    for ag in agents:
        set_seed(seed)
        h = eng.run(ag, log=False)
        out[ag.name] = summarize(h, cfg, event_round)
        curves[ag.name] = h["acc"]
    return out, curves, event_round


def aggregate(per_seed, seeds, scenario, drift_type, drift_mode="sudden",
              dropout_mode=None, dropout_rate=None):
    agents = list(per_seed[seeds[0]].keys())
    case = case_name(scenario, drift_type, drift_mode, dropout_mode, dropout_rate)
    primary = primary_metric_for(case)
    rows = []
    for ag in agents:
        row = {
            "case": case,
            "scenario": scenario,
            "drift_type": drift_type or "",
            "drift_mode": drift_mode if scenario == "drift" else "",
            "dropout_mode": dropout_mode if scenario == "dropout" else "",
            "dropout_rate": dropout_rate if scenario == "dropout" and dropout_rate is not None else "",
            "agent": ag,
        }
        for metric in METRICS:
            vals = [
                per_seed[seed][ag][metric]
                for seed in seeds
                if isinstance(per_seed[seed][ag][metric], (int, float))
            ]
            if vals:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 4)
                row[f"{metric}_std"] = (
                    round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
                )
            else:
                row[f"{metric}_mean"] = "n/a"
                row[f"{metric}_std"] = "n/a"
        row["tool"] = per_seed[seeds[0]][ag]["tool"]
        deltas = [
            per_seed[seed][ag][primary] - per_seed[seed]["noop"][primary]
            for seed in seeds
            if isinstance(per_seed[seed][ag][primary], (int, float))
            and isinstance(per_seed[seed]["noop"][primary], (int, float))
        ]
        row["primary_metric"] = primary
        row["primary_delta_vs_noop_mean"] = round(statistics.mean(deltas), 4) if deltas else "n/a"
        row["primary_delta_vs_noop_std"] = round(statistics.pstdev(deltas), 4) if len(deltas) > 1 else (0.0 if deltas else "n/a")
        row["primary_win_rate_vs_noop"] = round(sum(delta > 0 for delta in deltas) / len(deltas), 4) if deltas else "n/a"
        rows.append(row)
    return rows


def build_acceptance_rows(rows):
    reports = []
    for case in dict.fromkeys(row["case"] for row in rows):
        by_agent = {row["agent"]: row for row in rows if row["case"] == case}
        target_name = target_agent_for(case)
        if not target_name or any(name not in by_agent for name in ("noop", "random", "oracle", target_name)):
            reports.append({"case": case, "status": "n/a", "reason": "required agent missing"})
            continue

        target = by_agent[target_name]
        oracle = by_agent["oracle"]
        noop = by_agent["noop"]
        random = by_agent["random"]
        metric = target["primary_metric"]
        metric_col = f"{metric}_mean"
        excluded = {target_name, "oracle"} | PERFORMANCE_BOUND_AGENTS.get(case, set())
        bad = [row for row in by_agent.values()
               if row["agent"] not in excluded
               and (row["agent"] in ("noop", "random")
                    or as_float(row["misdiagnosis_penalty_mean"]) > 0)]
        bad = [row for row in bad if not math.isnan(as_float(row[metric_col]))]
        best_bad = max(bad, key=lambda row: as_float(row[metric_col]))
        target_score = as_float(target[metric_col])
        oracle_score = as_float(oracle[metric_col])
        best_bad_score = as_float(best_bad[metric_col])
        gap = target_score - best_bad_score
        oracle_consistent = oracle_score >= best_bad_score and oracle_score >= target_score - 0.01
        post_delta = as_float(target["post_drift_mean_acc_mean"]) - as_float(noop["post_drift_mean_acc_mean"])
        penalty_ok = as_float(target["misdiagnosis_penalty_mean"]) == 0
        structural_ok = True
        mechanism_ok = True
        oracle_gain = math.nan
        if case == "fault":
            mechanism_ok = as_float(target.get("switch_update_norm_ratio_mean")) >= 2.0
        elif case == "hetero":
            mechanism_ok = as_float(target.get("selected_action_rate_mean")) >= 0.80
        if case == "staggered":
            oracle_gain = post_delta / max(
                as_float(oracle["post_drift_mean_acc_mean"]) - as_float(noop["post_drift_mean_acc_mean"]),
                1e-12,
            )
            structural_ok = (
                post_delta >= 0.05
                and as_float(target["final_acc_mean"]) - as_float(noop["final_acc_mean"]) >= 0.08
                and oracle_gain >= 0.70
                and as_float(target["feddrift_cluster_purity_mean"]) >= 0.70
            )

        eligible = (penalty_ok and post_delta >= -0.01 and structural_ok
                    and mechanism_ok and oracle_consistent)
        status = "strong" if eligible and gap >= 0.02 else "conditional" if eligible and gap >= 0.01 else "fail"
        reports.append({
            "case": case,
            "primary_metric": metric,
            "target_agent": target_name,
            "target_score": round(target_score, 4),
            "oracle_score": round(oracle_score, 4),
            "best_bad_agent": best_bad["agent"],
            "best_bad_score": round(as_float(best_bad[metric_col]), 4),
            "gap": round(gap, 4),
            "target_delta_vs_noop": round(post_delta, 4),
            "random_delta_vs_noop": round(
                as_float(random[metric_col]) - as_float(noop[metric_col]), 4
            ),
            "oracle_gain": round(oracle_gain, 4) if not math.isnan(oracle_gain) else "n/a",
            "target_penalty": target["misdiagnosis_penalty_mean"],
            "over_intervention": target.get("over_intervention_mean", "n/a"),
            "switch_regret": target.get("switch_regret_mean", "n/a"),
            "recovery_rounds": target.get("recovery_rounds_mean", "n/a"),
            "switch_update_norm_ratio": target.get("switch_update_norm_ratio_mean", "n/a"),
            "selected_action_rate": target.get("selected_action_rate_mean", "n/a"),
            "feddrift_num_concepts": target.get("feddrift_num_concepts_mean", "n/a"),
            "mechanism_ok": mechanism_ok,
            "oracle_consistent": oracle_consistent,
            "status": status,
            "reason": "good/bad gap and guardrails" if eligible else "gap or guardrail failed",
        })
    return reports


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_run_metadata(args, out_dir):
    def git(*parts):
        result = subprocess.run(
            ["git", *parts], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    cases = [case_name(scenario, drift_type, args.drift_mode,
                       args.dropout_mode, args.dropout_rate)
             for scenario, drift_type in selected_cases(args)]
    git_status = git("status", "--porcelain")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cases": cases,
        "seeds": args.seeds,
        "arguments": vars(args),
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": None if git_status == "unavailable" else bool(git_status),
        "git_status": git_status,
        "git_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD").encode()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    path = os.path.join(out_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path


def plot_recovery_curves(per_seed, seeds, case, event_round, out_dir):
    if Image is None:
        print("  Pillow unavailable; skipped recovery curve")
        return None
    width, height = 1000, 560
    left, top, right, bottom = 62, 52, 220, 48
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    agents = list(per_seed[seeds[0]])
    palette = [(31, 119, 180), (188, 189, 34), (214, 39, 40), (44, 160, 44),
               (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207),
               (255, 127, 14), (17, 95, 55), (127, 127, 127), (0, 0, 0),
               (141, 211, 199), (190, 186, 218), (251, 128, 114)]
    rounds = len(per_seed[seeds[0]][agents[0]])

    draw.text((left, 18), f"DynFL-Bench B1 - {case} recovery (mean over {len(seeds)} seeds)",
              fill=(20, 20, 20), font=font)
    for tick in range(6):
        value = tick / 5
        y = top + plot_h - round(value * plot_h)
        draw.line((left, y, left + plot_w, y), fill=(225, 225, 225))
        draw.text((18, y - 5), f"{value:.1f}", fill=(60, 60, 60), font=font)
    draw.line((left, top, left, top + plot_h), fill=(50, 50, 50))
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(50, 50, 50))
    if case != "hetero" and rounds > 1:
        x = left + round(event_round * plot_w / (rounds - 1))
        draw.line((x, top, x, top + plot_h), fill=(110, 110, 110))
        draw.text((x + 4, top + 4), f"event r{event_round}", fill=(80, 80, 80), font=font)

    for index, agent in enumerate(agents):
        curves = [per_seed[seed][agent] for seed in seeds]
        mean_curve = [statistics.mean(values) for values in zip(*curves)]
        color = palette[index % len(palette)]
        points = [(left + round(r * plot_w / max(rounds - 1, 1)),
                   top + plot_h - round(max(0.0, min(1.0, value)) * plot_h))
                  for r, value in enumerate(mean_curve)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        legend_y = top + index * 29
        draw.line((left + plot_w + 22, legend_y + 5, left + plot_w + 48, legend_y + 5),
                  fill=color, width=3)
        draw.text((left + plot_w + 56, legend_y), agent, fill=(35, 35, 35), font=font)
    draw.text((left + plot_w // 2 - 18, height - 25), "round", fill=(50, 50, 50), font=font)
    draw.text((8, top - 20), "accuracy", fill=(50, 50, 50), font=font)
    path = os.path.join(out_dir, f"recovery_{case}_multiseed.png")
    image.save(path)
    return path


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def gradient_color(value, palette):
    value = max(0.0, min(1.0, value))
    scaled = value * (len(palette) - 1)
    idx = min(int(scaled), len(palette) - 2)
    frac = scaled - idx
    a = palette[idx]
    b = palette[idx + 1]
    return tuple(round(a[channel] + (b[channel] - a[channel]) * frac) for channel in range(3))


def heat_color(value, vmin, vmax, cmap):
    if math.isnan(value):
        return (235, 235, 235)
    if vmax <= vmin:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
    palettes = {
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        "magma": [(0, 0, 4), (73, 15, 111), (182, 54, 121), (251, 136, 97), (252, 253, 191)],
        "diverging": [(49, 54, 149), (116, 173, 209), (247, 247, 247), (244, 109, 67), (165, 0, 38)],
    }
    return gradient_color(norm, palettes[cmap])


def plot_heatmap(rows, metric, out_dir, filename, title, cmap, vmin=None, vmax=None):
    cases = list(dict.fromkeys(row["case"] for row in rows))
    agents = list(dict.fromkeys(row["agent"] for row in rows))
    lookup = {(row["case"], row["agent"]): as_float(row.get(metric)) for row in rows}
    matrix = [
        [lookup.get((case, agent), math.nan) for case in cases]
        for agent in agents
    ]

    finite = [value for row in matrix for value in row if not math.isnan(value)]
    if not finite:
        finite = [0.0]
    vmin = min(finite) if vmin is None else vmin
    vmax = max(finite) if vmax is None else vmax

    font = ImageFont.load_default()
    cell_w = 96
    cell_h = 30
    left = 190
    top = 74
    right = 98
    bottom = 42
    width = max(640, left + cell_w * len(cases) + right)
    height = top + cell_h * len(agents) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((left, 18), title, fill=(20, 20, 20), font=font)
    for j, case in enumerate(cases):
        draw.text((left + j * cell_w + 4, top - 22), case, fill=(35, 35, 35), font=font)
    for i, agent in enumerate(agents):
        y = top + i * cell_h + 8
        draw.text((8, y), agent, fill=(35, 35, 35), font=font)

    for i, agent in enumerate(agents):
        for j, case in enumerate(cases):
            value = matrix[i][j]
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            color = heat_color(value, vmin, vmax, cmap)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255))
            label = "n/a" if math.isnan(value) else f"{value:.{3 if 'delta' in metric else 2}f}"
            tw, th = text_size(draw, label, font)
            text_color = (255, 255, 255) if sum(color) < 360 else (15, 15, 15)
            draw.text((x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2), label, fill=text_color, font=font)

    bar_x = left + cell_w * len(cases) + 24
    bar_y = top
    bar_h = max(1, cell_h * len(agents))
    for offset in range(bar_h):
        norm = 1.0 - offset / max(1, bar_h - 1)
        draw.line(
            [(bar_x, bar_y + offset), (bar_x + 14, bar_y + offset)],
            fill=heat_color(vmin + norm * (vmax - vmin), vmin, vmax, cmap),
        )
    draw.rectangle([bar_x, bar_y, bar_x + 14, bar_y + bar_h], outline=(90, 90, 90))
    draw.text((bar_x + 20, bar_y - 4), f"{vmax:.2f}", fill=(35, 35, 35), font=font)
    draw.text((bar_x + 20, bar_y + bar_h - 8), f"{vmin:.2f}", fill=(35, 35, 35), font=font)

    path = os.path.join(out_dir, filename)
    image.save(path)
    return path


def plot_summary(rows, out_dir):
    if Image is None:
        print("  Pillow unavailable; skipped PNG plots")
        return []

    max_delta = max(0.01, max([abs(as_float(row["primary_delta_vs_noop_mean"])) for row in rows
                               if not math.isnan(as_float(row["primary_delta_vs_noop_mean"]))] or [0.01]))
    return [
        plot_heatmap(
            rows,
            "primary_delta_vs_noop_mean",
            out_dir,
            "b1_delta_vs_noop_heatmap.png",
            "Paired primary-metric delta vs noop (higher is better)",
            "diverging",
            vmin=-max_delta,
            vmax=max_delta,
        ),
        plot_heatmap(
            rows,
            "post_drift_mean_acc_mean",
            out_dir,
            "b1_post_mean_heatmap.png",
            "Post-event mean accuracy (higher is better)",
            "viridis",
            vmin=0.0,
            vmax=1.0,
        ),
        plot_heatmap(
            rows,
            "misdiagnosis_penalty_mean",
            out_dir,
            "b1_penalty_heatmap.png",
            "Misdiagnosis penalty (lower is better)",
            "magma",
        ),
        plot_heatmap(
            rows,
            "post_event_worst_acc_mean",
            out_dir,
            "b1_worst_class_heatmap.png",
            "Post-event worst-class accuracy (higher is better)",
            "viridis",
            vmin=0.0,
            vmax=1.0,
        ),
        plot_heatmap(
            rows,
            "feddrift_cluster_purity_mean",
            out_dir,
            "b1_cluster_purity_heatmap.png",
            "FedDrift final cluster purity (higher is better)",
            "viridis",
            vmin=0.0,
            vmax=1.0,
        ),
        plot_heatmap(
            rows,
            "over_intervention_mean",
            out_dir,
            "b1_over_intervention_heatmap.png",
            "Pre-event active interventions (lower is better)",
            "magma",
            vmin=0.0,
        ),
    ]


def write_action_confusions(decisions, out_dir):
    fields = ["agent", "true_case", "chosen_action", "count", "rate"]
    rows, paths = [], []
    for agent in ("diagnose", "llm"):
        selected = [row for row in decisions if row["agent"] == agent]
        if not selected:
            continue
        cases = list(dict.fromkeys(row["case"] for row in selected))
        actions = sorted({action for row in selected
                          for action in row["action_sequence"].split("|")})
        heat_rows = []
        for action in actions:
            for case in cases:
                case_rows = [row for row in selected if row["case"] == case]
                count = sum(action in row["action_sequence"].split("|") for row in case_rows)
                rate = count / len(case_rows) if case_rows else 0.0
                rows.append({"agent": agent, "true_case": case,
                             "chosen_action": action, "count": count, "rate": round(rate, 4)})
                heat_rows.append({"case": case, "agent": action, "rate": rate})
        if Image is not None:
            paths.append(plot_heatmap(
                heat_rows, "rate", out_dir, f"action_confusion_{agent}.png",
                f"{agent} action-sequence confusion (columns=true case)", "viridis", 0.0, 1.0
            ))
    path = os.path.join(out_dir, "action_confusion.csv")
    write_csv(path, fields, rows)
    return [path] + paths


def write_acceptance_report(rows, out_dir):
    report = build_acceptance_rows(rows)
    fields = ["case", "primary_metric", "target_agent", "target_score", "oracle_score",
              "best_bad_agent", "best_bad_score", "gap", "target_delta_vs_noop",
              "random_delta_vs_noop", "oracle_gain", "target_penalty", "over_intervention",
              "switch_regret", "recovery_rounds", "switch_update_norm_ratio",
              "selected_action_rate", "feddrift_num_concepts", "mechanism_ok",
              "oracle_consistent", "status", "reason"]
    csv_path = os.path.join(out_dir, "b1_acceptance_report.csv")
    write_csv(csv_path, fields, report)
    md_path = os.path.join(out_dir, "b1_acceptance_report.md")
    with open(md_path, "w") as f:
        f.write("# B1 acceptance report\n\n")
        f.write("| case | target | primary | gap | random-noop | status |\n")
        f.write("|---|---|---|---:|---:|---|\n")
        for row in report:
            f.write(f"| {row['case']} | {row.get('target_agent', '')} | {row.get('primary_metric', '')} "
                    f"| {row.get('gap', '')} | {row.get('random_delta_vs_noop', '')} | {row['status']} |\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--drift_types",
        nargs="+",
        choices=DRIFT_TYPES,
        help="Drift-source variants to run when scenario=drift.",
    )
    parser.add_argument(
        "--b1_full",
        action="store_true",
        help="Run drift real/virtual/label plus fault/dropout/hetero/staggered.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--drift_mode", choices=["sudden", "gradual", "recurrent"], default="sudden")
    parser.add_argument("--virtual_shift", type=float)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal_evidence", action="store_true")
    parser.add_argument("--proxy_evidence", action="store_true")
    parser.add_argument("--dropout_mode", choices=["correlated", "random", "fdms_clustered"])
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--fedlc_tau", type=float)
    parser.add_argument("--feddaa_T", type=int)
    parser.add_argument("--label_imbalance_factor", type=float)
    parser.add_argument("--label_min_per_class", type=int)
    parser.add_argument("--label_prior_strength", type=float)
    parser.add_argument("--decide_every", type=int)
    parser.add_argument("--agent_names", nargs="+")
    parser.add_argument("--llm_model")
    parser.add_argument("--llm_base_url")
    parser.add_argument("--llm_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_reasoning_effort")
    parser.add_argument("--llm_input_cost_per_million", type=float, default=0.0)
    parser.add_argument("--llm_output_cost_per_million", type=float, default=0.0)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--partition_mode", choices=PARTITION_MODES, default="noniid")
    args = parser.parse_args()
    if args.dropout_rate is not None and not 0.0 <= args.dropout_rate < 1.0:
        parser.error("--dropout_rate must be in [0, 1)")
    if args.decide_every is not None and args.decide_every < 1:
        parser.error("--decide_every must be >= 1")
    if args.feddaa_T is not None and args.feddaa_T < 1:
        parser.error("--feddaa_T must be >= 1")
    if args.agent_names and "noop" not in args.agent_names:
        parser.error("--agent_names must include noop for paired deltas")
    if args.llm_input_cost_per_million < 0 or args.llm_output_cost_per_million < 0:
        parser.error("LLM token prices must be non-negative")
    llm_settings = None
    if args.llm_model:
        llm_settings = {
            "model": args.llm_model,
            "base_url": args.llm_base_url,
            "api_key_env": args.llm_api_key_env,
            "temperature": args.llm_temperature,
            "reasoning_effort": args.llm_reasoning_effort,
            "input_cost_per_million": args.llm_input_cost_per_million,
            "output_cost_per_million": args.llm_output_cost_per_million,
        }

    args.out_dir = str(partition_output_dir(args.out_dir, args.partition_mode))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"  saved {write_run_metadata(args, args.out_dir)}")
    fields = ["partition_mode", "case", "scenario", "drift_type", "drift_mode", "dropout_mode",
              "dropout_rate", "agent", "tool", "primary_metric",
              "primary_delta_vs_noop_mean", "primary_delta_vs_noop_std",
              "primary_win_rate_vs_noop"] + [
        f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")
    ]

    all_rows = []
    decisions = []
    decision_trace = []
    for scenario, drift_type in selected_cases(args):
        label = case_name(scenario, drift_type, args.drift_mode,
                          args.dropout_mode, args.dropout_rate)
        drift_text = f" drift_type={drift_type} drift_mode={args.drift_mode}" if drift_type else ""
        print(f"\n===== case={label} scenario={scenario}{drift_text} seeds={args.seeds} =====")
        per_seed = {}
        per_seed_curves = {}
        event_round = None
        for seed in args.seeds:
            print(f"  seed {seed} ...", flush=True)
            per_seed[seed], per_seed_curves[seed], event_round = run_one(
                scenario=scenario, seed=seed, synthetic=args.synthetic, smoke=args.smoke,
                dataset=args.dataset,
                formal_evidence=args.formal_evidence, proxy_evidence=args.proxy_evidence,
                drift_type=drift_type, virtual_shift=args.virtual_shift,
                dropout_mode=args.dropout_mode,
                dropout_rate=args.dropout_rate, fedlc_tau=args.fedlc_tau,
                feddaa_T=args.feddaa_T,
                label_imbalance_factor=args.label_imbalance_factor,
                label_min_per_class=args.label_min_per_class,
                label_prior_strength=args.label_prior_strength, out_dir=args.out_dir,
                drift_mode=args.drift_mode, decide_every=args.decide_every,
                agent_names=args.agent_names,
                llm_settings=llm_settings,
                partition_mode=args.partition_mode,
            )

        rows = aggregate(per_seed, args.seeds, scenario, drift_type, args.drift_mode,
                         args.dropout_mode, args.dropout_rate)
        for row in rows:
            row["partition_mode"] = args.partition_mode
        for seed in args.seeds:
            for agent, summary in per_seed[seed].items():
                decisions.append({
                    "partition_mode": args.partition_mode,
                    "case": label,
                    "seed": seed,
                    "agent": agent,
                    "tool": summary["tool"],
                    "action_sequence": summary["action_sequence"],
                    "active_action_count": summary["active_action_count"],
                    "wrong_action_count": summary["wrong_action_count"],
                    "misdiagnosis_penalty": summary["misdiagnosis_penalty"],
                    "primary_score": summary[primary_metric_for(label)],
                    "switch_round": summary["switch_round"],
                    "switch_regret": summary["switch_regret"],
                    "llm_calls": summary["llm_calls"],
                    "llm_total_tokens": summary["llm_total_tokens"],
                    "llm_cost_usd": summary["llm_cost_usd"],
                    "llm_latency_ms_mean": summary["llm_latency_ms_mean"],
                    "llm_latency_ms_p95": summary["llm_latency_ms_p95"],
                })
                for event in summary["decision_trace"]:
                    decision_trace.append({"partition_mode": args.partition_mode,
                                           "case": label, "seed": seed,
                                           "agent": agent, **event})
        all_rows.extend(rows)
        path = os.path.join(args.out_dir, f"metrics_{label}_multiseed.csv")
        write_csv(path, fields, rows)
        print(f"  saved {path}")
        curve_path = plot_recovery_curves(
            per_seed_curves, args.seeds, label, event_round, args.out_dir
        )
        if curve_path:
            print(f"  saved {curve_path}")
        for row in rows:
            post_mean = row["post_drift_mean_acc_mean"]
            post_std = row["post_drift_mean_acc_std"]
            worst_mean = row["post_event_worst_acc_mean"]
            penalty = row["misdiagnosis_penalty_mean"]
            print(
                f"    {row['agent']:22s} post_mean={post_mean}+/-{post_std} "
                f"worst={worst_mean} penalty={penalty} tool={row['tool']}"
            )

    combined = os.path.join(args.out_dir, "metrics_all_multiseed.csv")
    write_csv(combined, fields, all_rows)
    print(f"\n  saved {combined}")
    decision_path = os.path.join(args.out_dir, "decisions_per_seed.csv")
    decision_fields = ["partition_mode", "case", "seed", "agent", "tool", "action_sequence",
                       "active_action_count", "wrong_action_count", "misdiagnosis_penalty",
                       "primary_score", "switch_round", "switch_regret", "llm_calls",
                       "llm_total_tokens", "llm_cost_usd", "llm_latency_ms_mean",
                       "llm_latency_ms_p95"]
    write_csv(decision_path, decision_fields, decisions)
    print(f"  saved {decision_path}")
    trace_path = os.path.join(args.out_dir, "decision_trace.csv")
    trace_fields = ["partition_mode", "case", "seed", "agent", "round", "action", "reasoning",
                    "diagnosis", "llm_called", "llm_model", "llm_latency_ms",
                    "llm_prompt_tokens", "llm_completion_tokens", "llm_cost_usd"]
    write_csv(trace_path, trace_fields, decision_trace)
    print(f"  saved {trace_path}")
    timing = os.path.join(args.out_dir, "b1_decision_timing.csv")
    timing_fields = ["case", "agent", "tool", "over_intervention_mean", "switch_round_mean",
                     "trigger_target_round_mean", "switch_regret_mean", "switch_update_norm_ratio_mean",
                     "selected_action_rate_mean", "recovery_rounds_mean",
                     "feddrift_first_trigger_round_mean"]
    write_csv(timing, timing_fields, all_rows)
    print(f"  saved {timing}")
    for path in write_acceptance_report(all_rows, args.out_dir):
        print(f"  saved {path}")
    for path in plot_summary(all_rows, args.out_dir):
        print(f"  saved {path}")
    for path in write_action_confusions(decisions, args.out_dir):
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
