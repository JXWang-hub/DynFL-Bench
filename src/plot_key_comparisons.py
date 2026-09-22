"""Plot compact per-scenario comparisons from a completed run directory."""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


KEY_AGENTS = {
    "drift_cifar10_sudden_real": ("noop", "adapt_lr", "diagnose", "oracle", "reject_robust"),
    "drift_cifar10_sudden_virtual": ("noop", "diagnose", "oracle", "adapt_lr"),
    "drift_cifar10_sudden_label": ("noop", "label_prior_adapt", "reweight", "oracle", "adapt_lr"),
    "dropout_cifar10_correlated": ("noop", "handle_dropout", "friend_substitute", "oracle", "llm"),
    "dropout_cifar10_random": ("noop", "handle_dropout", "friend_substitute", "oracle", "llm"),
    "fault_cifar10": ("noop", "reject_robust", "oracle", "diagnose", "random"),
    "staggered_cifar10": ("noop", "spawn_concept", "spawn_concept_auto", "oracle", "llm"),
    "hetero_cifar10": ("noop", "switch_prox", "select_clients", "oracle", "reweight"),
}

COLORS = {
    "noop": "#555555",
    "oracle": "#111111",
    "diagnose": "#1f77b4",
    "llm": "#9467bd",
    "adapt_lr": "#2ca02c",
    "reject_robust": "#d62728",
    "handle_dropout": "#ff7f0e",
    "friend_substitute": "#8c564b",
    "spawn_concept": "#17becf",
    "spawn_concept_auto": "#bcbd22",
    "switch_prox": "#7f7f7f",
    "select_clients": "#e377c2",
    "label_prior_adapt": "#1f9d55",
    "reweight": "#66a61e",
    "random": "#999999",
}


def read_curve(path: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["round"]))
            ys.append(float(row["global_acc"]))
    return xs, ys


def tag_from_metrics(path: str) -> str:
    name = os.path.basename(path)
    return name.removeprefix("metrics_").removesuffix(".csv")


def choose_agents(tag: str) -> tuple[str, ...]:
    for prefix, agents in KEY_AGENTS.items():
        if tag.startswith(prefix):
            return agents
    return ("noop", "oracle")


def title_for(tag: str) -> str:
    return tag.replace("_proxy_T80_s0", "").replace("_", " ")


def plot_one(run_dir: str, out_dir: str, metrics_path: str) -> str:
    tag = tag_from_metrics(metrics_path)
    agents = choose_agents(tag)
    plt.figure(figsize=(8, 4.8))
    plotted = 0
    for agent in agents:
        tpath = os.path.join(run_dir, f"telemetry_{tag}_{agent}.csv")
        if not os.path.exists(tpath):
            continue
        xs, ys = read_curve(tpath)
        lw = 2.6 if agent in ("oracle", "select_clients", "handle_dropout", "spawn_concept") else 1.9
        ls = "--" if agent in ("llm", "diagnose", "random") else "-"
        plt.plot(xs, ys, label=agent, color=COLORS.get(agent), linewidth=lw, linestyle=ls)
        plotted += 1

    if plotted == 0:
        return ""
    if "hetero" not in tag:
        plt.axvline(40, color="#777777", linestyle=":", linewidth=1.2, label="event @ 40")
        if "staggered" in tag:
            plt.axvline(60, color="#999999", linestyle=":", linewidth=1.2, label="event @ 60")
    plt.title(title_for(tag), fontsize=11)
    plt.xlabel("round")
    plt.ylabel("global accuracy")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(out_dir, f"key_{tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/runs/b2_cifar10_single_20260701_171954")
    ap.add_argument("--out_dir")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join("results", "key_plots_" + os.path.basename(args.run_dir))
    os.makedirs(out_dir, exist_ok=True)
    outs = [
        plot_one(args.run_dir, out_dir, path)
        for path in sorted(glob.glob(os.path.join(args.run_dir, "metrics_*.csv")))
    ]
    outs = [p for p in outs if p]
    print(f"saved {len(outs)} plots to {out_dir}")
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
