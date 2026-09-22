"""B1 deliverable: the diagnosis->act separation triptych.

Reads the already-written results/telemetry_{scenario}_{agent}.csv (per-round
global_acc + action) and results/metrics_{scenario}.csv (tool_matched), and
draws drift | fault | dropout as one row of three panels. The same agent keeps
one colour across panels so the role reversal (an agent that wins its own
scenario垫底 in the others) is visible at a glance.

  python plot_triptych.py                 # auto-discover available result tags
  python plot_triptych.py --scenarios drift_sudden_real_r40_T80_s0 fault_r40_T80_s0

Needs matplotlib. If the local install is broken (missing DLL, see 进展 §10),
run `conda install -n dynfl matplotlib` or run this in the cloud env.
"""
import argparse, csv, glob, os

BASE_SCENARIOS = ["drift", "fault", "dropout", "hetero", "staggered"]

# fixed agent -> colour so the same line is the same colour in every panel
COLORS = {
    "noop":           "#9e9e9e",
    "random":         "#bcbd22",
    "adapt_lr":       "#1f77b4",
    "reject_robust":  "#d62728",
    "handle_dropout": "#2ca02c",
    "switch_prox":    "#9467bd",
    "select_clients": "#e377c2",
    "label_prior_adapt": "#1f9d55",
    "spawn_concept":  "#17becf",
    "spawn_concept_auto": "#8c564b",
    "oracle":         "#000000",
    "llm":            "#ff7f0e",
}
ORDER = ["noop", "random", "adapt_lr", "reject_robust", "handle_dropout",
         "friend_substitute", "switch_prox", "spawn_concept", "spawn_concept_auto",
         "label_prior_adapt", "select_clients", "oracle", "llm"]

YLAB = {
    "drift":   "acc on NEW concept (follow drift)",
    "fault":   "acc on original task (reject noise)",
    "dropout": "acc on original task (despite dropout)",
    "hetero":  "acc on original task (extreme non-IID)",
    "staggered": "mean acc across concepts",
}
CORRECT_TOOL = {"drift": "drift_adapt/moment_align_adapt/label_prior_adapt", "fault": "robust",
                "dropout": "friend_substitute", "hetero": "fedprox",
                "staggered": "spawn_concept_auto"}


def read_accs(scenario, agent):
    path = f"results/telemetry_{scenario}_{agent}.csv"
    if not os.path.exists(path):
        return None
    accs = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            accs.append(float(row["global_acc"]))
    return accs


def read_matched(scenario):
    """agent -> 'yes'/'no' from metrics_{scenario}.csv."""
    path = f"results/metrics_{scenario}.csv"
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                out[row["agent"]] = row.get("tool_matched", "")
    return out


def read_correct_tool(scenario):
    path = f"results/metrics_{scenario}.csv"
    if os.path.exists(path):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                tool = row.get("formal_correct_tool", "")
                if tool:
                    return tool
                break
    return CORRECT_TOOL.get(base_scenario(scenario), "?")


def agents_present(scenario):
    found = []
    for f in glob.glob(f"results/telemetry_{scenario}_*.csv"):
        stem = os.path.basename(f)[:-4]
        for agent in ORDER:
            if stem.endswith(f"_{agent}"):
                found.append(agent)
                break
    return [a for a in ORDER if a in found] + [a for a in found if a not in ORDER]


def base_scenario(tag):
    return tag.split("_", 1)[0]


def discover_tags():
    tags = set()
    for f in glob.glob("results/telemetry_*.csv"):
        stem = os.path.basename(f)[len("telemetry_"):-4]
        for agent in ORDER:
            if stem.endswith(f"_{agent}"):
                tags.add(stem[:-(len(agent) + 1)])
                break
    return sorted(tags)


def resolve_tags(requested):
    found = discover_tags()
    if requested:
        out = []
        for item in requested:
            if item in found:
                out.append(item)
            elif item in BASE_SCENARIOS:
                out.extend(t for t in found if base_scenario(t) == item or t == item)
            else:
                out.append(item)
        return [t for t in out if agents_present(t)]
    out = []
    for base in BASE_SCENARIOS:
        exact = [t for t in found if t == base]
        tagged = [t for t in found if base_scenario(t) == base]
        if exact:
            out.extend(exact)
        elif tagged:
            out.append(tagged[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, default=40, help="round of the disturbance")
    ap.add_argument("--out", help="output path; default is based on selected result tags")
    ap.add_argument("--scenarios", nargs="+", help="result tags or base scenarios to plot")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scen = resolve_tags(args.scenarios)
    if not scen:
        raise SystemExit("no telemetry CSVs found in results/ — run run.py first")

    fig, axes = plt.subplots(1, len(scen), figsize=(5.0 * len(scen), 4.6), sharey=False)
    if len(scen) == 1:
        axes = [axes]

    for ax, s in zip(axes, scen):
        matched = read_matched(s)
        for agent in agents_present(s):
            accs = read_accs(s, agent)
            if not accs:
                continue
            m = matched.get(agent, "")
            tag = " (match)" if m == "yes" else (" (wrong)" if m == "no" else "")
            ax.plot(accs, label=f"{agent}{tag}", color=COLORS.get(agent),
                    linewidth=2.0 if m == "yes" else 1.4,
                    alpha=1.0 if m == "yes" else 0.85)
        base = base_scenario(s)
        if base != "hetero":                       # hetero is static -> no event line
            ax.axvline(args.event, color="gray", ls=":", linewidth=1.2)
        ax.set_title(f"{s}  (correct: {read_correct_tool(s)})", fontsize=11)
        ax.set_xlabel("round")
        ax.set_ylabel(YLAB.get(base, "accuracy"), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("DynFL-Bench B1 — diagnosis->act: same agents, role reversal across scenarios",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs("results", exist_ok=True)
    out = args.out or f"results/triptych_{'_'.join(scen)}.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}  ({len(scen)} panels: {', '.join(scen)})")


if __name__ == "__main__":
    main()
