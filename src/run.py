"""Entry point (B1 + dropout). Runs all agents on one scenario, plots the 对症
recovery curve, dumps metrics + per-agent telemetry.

  python run.py --scenario drift      # genuine drift  -> lr_reset wins
  python run.py --scenario fault      # faulty clients -> robust wins
  python run.py --scenario dropout    # client dropout -> dropout_handle wins
"""
import argparse, csv, os
import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False
    print(f"[warn] matplotlib unavailable ({e}); skipping figure, CSVs still written.")

from config import (Config, DATASETS, DRIFT_FRACTION, PARTITION_MODES, partition_output_dir,
                    validate_partition_mode)
from data import (load_data, dirichlet_partition, iid_partition, make_drift_map, make_drift_maps,
                  apply_virtual_shift, partition_sha256, partition_without_replacement,
                  resample_label_longtail)
from engine import Engine
from scoring import summarize
from agents import make_agents
from evidence import write_markdown


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _tag_num(x):
    return str(x).replace(".", "p")


def _fdms_clustered_partition(y, cfg, n_classes):
    clients = list(range(cfg.num_clients))
    groups = [g.tolist() for g in np.array_split(clients, min(cfg.fdms_clusters, cfg.num_clients))]
    probs = np.zeros((cfg.num_clients, n_classes), dtype=float)
    for gi, group in enumerate(groups):
        labels = [(2 * gi) % n_classes, (2 * gi + 1) % n_classes]
        for ci in group:
            probs[ci, labels] = 1.0
    parts = partition_without_replacement(
        y, probs, cfg.samples_per_client, cfg.seed + 3000
    )
    return parts, groups


def output_tag(cfg):
    """Stable filename tag for a run. Include the knobs that change results so
    sub-scenarios (drift real/virtual/label, sudden/gradual, seeds, etc.) do not
    overwrite each other."""
    parts = [cfg.scenario, cfg.dataset, validate_partition_mode(cfg.partition_mode)]
    if cfg.scenario == "drift":
        parts += [cfg.drift_mode, cfg.drift_type, f"r{cfg.drift_round}"]
        if cfg.drift_type == "virtual":
            parts.append(f"v{_tag_num(cfg.virtual_shift)}")
        elif cfg.drift_type == "label":
            parts.append(f"lt{_tag_num(cfg.label_imbalance_factor)}")
            if float(getattr(cfg, "fedlc_tau", 1.0)) != 1.0:
                parts.append(f"tau{_tag_num(cfg.fedlc_tau)}")
        if cfg.drift_type in ("real", "virtual") and int(cfg.feddaa_T) != 6:
            parts.append(f"daaT{cfg.feddaa_T}")
        if cfg.drift_mode == "recurrent":
            parts.append(f"back{cfg.drift_round + cfg.recur_gap}")
    elif cfg.scenario == "dropout":
        parts += [cfg.dropout_mode, f"r{cfg.drift_round}"]
        if cfg.dropout_mode in ("random", "fdms_clustered"):
            parts.append(f"a{int(cfg.dropout_rate * 100)}")
        if cfg.dropout_recover_round > 0:
            parts.append(f"rec{cfg.dropout_recover_round}")
    elif cfg.scenario == "hetero":
        parts += [f"b{_tag_num(cfg.hetero_beta)}", f"e{cfg.local_epochs}",
                  f"mu{_tag_num(cfg.fedprox_mu)}", f"p{int(cfg.participation * 100)}"]
    elif cfg.scenario == "staggered":
        parts += [f"k{cfg.n_concepts}", f"gap{cfg.stagger_gap}", f"r{cfg.drift_round}"]
    elif cfg.scenario == "fault":
        parts += [f"r{cfg.drift_round}", cfg.robust_variant,
                  f"c{int(cfg.byzantine_frac * 100)}",
                  f"x{_tag_num(cfg.byzantine_scale)}"]
    if cfg.scenario != "hetero" and cfg.participation < 1.0:
        parts.append(f"p{int(cfg.participation * 100)}")
    if cfg.decide_every != 1:
        parts.append(f"d{cfg.decide_every}")
    if getattr(cfg, "formal_evidence", False):
        parts.append("formal")
    if getattr(cfg, "proxy_evidence", False):
        parts.append("proxy")
    parts += [f"T{cfg.rounds}", f"s{cfg.seed}"]
    return "_".join(parts)


def build_data(cfg):
    (Xtr, ytr), (Xte, yte), n_classes, input_shape = load_data(cfg)
    b3_causes = set(getattr(cfg, "b3_active_causes", ()))
    part_beta = (cfg.hetero_beta if ("partial_participation_hetero" in b3_causes or
                                     cfg.scenario == "hetero") else cfg.dirichlet_beta)
    fdms_case = ((cfg.scenario == "dropout" or "dropout" in b3_causes) and
                 cfg.dropout_mode == "fdms_clustered")
    fdms_groups = ([g.tolist() for g in np.array_split(
        list(range(cfg.num_clients)), min(cfg.fdms_clusters, cfg.num_clients)
    )] if fdms_case else [])
    mode = validate_partition_mode(cfg.partition_mode)
    if mode == "iid":
        parts = iid_partition(ytr, cfg.num_clients, cfg.samples_per_client, cfg.seed)
    elif fdms_case:
        parts, fdms_groups = _fdms_clustered_partition(ytr, cfg, n_classes)
    else:
        parts = dirichlet_partition(ytr, cfg.num_clients, part_beta,
                                    cfg.samples_per_client, cfg.seed)
    client_X = [Xtr[p] for p in parts]
    client_y = [ytr[p] for p in parts]

    # drift/fault: which clients relabel. dropout: none relabel.
    data_causes = {"real_drift", "virtual_drift", "label_prior_drift"}
    n_drift = (cfg.num_clients if b3_causes & data_causes else
               int(cfg.drift_fraction * cfg.num_clients))
    drift_clients = set(np.random.default_rng(cfg.seed + 7)
                        .choice(cfg.num_clients, n_drift, replace=False).tolist()) if n_drift else set()

    # flip schedule: ci -> round at which it starts using the drifted labels.
    # sudden = everyone flips at drift_round; gradual = waves of p% every K rounds.
    drift_schedule = {ci: cfg.drift_round for ci in drift_clients}
    if cfg.scenario == "drift" and cfg.drift_mode == "gradual":
        order = list(drift_clients)
        np.random.default_rng(cfg.seed + 9).shuffle(order)          # deterministic waves
        per_wave = max(1, int(round(cfg.drift_wave_frac * cfg.num_clients)))
        for w, start in enumerate(range(0, len(order), per_wave)):
            for ci in order[start:start + per_wave]:
                drift_schedule[ci] = cfg.drift_round + w * cfg.drift_wave_every

    # staggered (AC1.2.4): split clients into n_concepts clusters; cluster k drifts
    # to its OWN concept k at drift_round + (k-1)*stagger_gap (staggered in time+space).
    concept_of, drift_maps_multi = {}, {}
    if cfg.scenario == "staggered":
        drift_maps_multi = make_drift_maps(cfg.n_concepts, n_classes, cfg.seed)
        order = list(np.random.default_rng(cfg.seed + 11).permutation(cfg.num_clients))
        per = max(1, cfg.num_clients // cfg.n_concepts)
        for idx, ci in enumerate(order):
            k = min(cfg.n_concepts, idx // per + 1)         # cluster id 1..K
            concept_of[int(ci)] = k
            drift_schedule[int(ci)] = cfg.drift_round + (k - 1) * cfg.stagger_gap

    # dropout (correlated mode): a group whose dominant class is in the lower
    # half of the label space goes absent -> biases the global model. random
    # mode picks the absent set per-round in the engine, so nothing to precompute.
    dropped_clients = set()
    if (cfg.scenario == "dropout" or "dropout" in b3_causes) and cfg.dropout_mode == "correlated":
        half = n_classes // 2
        for ci, yc in enumerate(client_y):
            dom = int(torch.bincount(yc, minlength=n_classes).argmax())
            if dom < half:
                dropped_clients.add(ci)

    # drift SOURCE type (AC1.2.5): real = current behavior (flip labels via drift_map);
    # virtual = shifted inputs; label = resampled to skew the label marginal.
    # baseline input mean + label histogram (r0, no drift) feed input_shift/label_shift.
    x0_all = torch.cat(client_X).float()
    baseline_xmean = float(x0_all.mean())
    baseline_xstd = max(float(x0_all.std(unbiased=False)), 1e-6)
    all_y0 = torch.cat(client_y)
    baseline_label_hist = torch.bincount(all_y0, minlength=n_classes).float()
    baseline_label_hist = baseline_label_hist / baseline_label_hist.sum().clamp_min(1)
    client_label_hist = []
    for y in client_y:
        hist = torch.bincount(y, minlength=n_classes).float()
        client_label_hist.append(hist / hist.sum().clamp_min(1))
    vshift_X, vshift_Xte, label_X, label_y, label_Xte, label_yte = None, None, None, None, None, None
    vshift_xmean, vshift_xstd = baseline_xmean, baseline_xstd
    label_class_weight = torch.ones(n_classes, dtype=torch.float32)
    label_calibration = torch.zeros(n_classes, dtype=torch.float32)
    label_target_hist = baseline_label_hist
    label_prior_calibration = torch.zeros(n_classes, dtype=torch.float32)
    if ((cfg.scenario == "drift" and cfg.drift_type == "virtual") or
            "virtual_drift" in b3_causes):
        vshift_X = [apply_virtual_shift(x, cfg.virtual_shift) for x in client_X]
        vshift_Xte = apply_virtual_shift(Xte, cfg.virtual_shift)
        xv_all = torch.cat(vshift_X).float()
        vshift_xmean = float(xv_all.mean())
        vshift_xstd = max(float(xv_all.std(unbiased=False)), 1e-6)
    elif ((cfg.scenario == "drift" and cfg.drift_type == "label") or
          "label_prior_drift" in b3_causes):
        pairs = [resample_label_longtail(client_X[i], client_y[i],
                                         cfg.label_imbalance_factor, n_classes, cfg.seed,
                                         cfg.label_min_per_class)
                 for i in range(cfg.num_clients)]
        label_X = [p[0] for p in pairs]; label_y = [p[1] for p in pairs]
        label_Xte, label_yte = resample_label_longtail(Xte, yte,
                                                       cfg.label_imbalance_factor, n_classes, cfg.seed,
                                                       cfg.label_min_per_class)
        counts = torch.bincount(torch.cat(label_y), minlength=n_classes).float().clamp_min(1)
        label_target_hist = counts / counts.sum().clamp_min(1)
        label_class_weight = torch.sqrt(counts.sum() / (n_classes * counts))
        label_class_weight = label_class_weight / label_class_weight.mean()
        label_calibration = float(cfg.fedlc_tau) * counts.pow(-0.25)
        label_prior_calibration = float(cfg.label_prior_strength) * -torch.log(label_target_hist.clamp_min(1e-6))

    drift_map = make_drift_map(n_classes, cfg.seed)
    return {"client_X": client_X, "client_y": client_y,
            "client_indices": parts, "partition_sha256": partition_sha256(parts, mode),
            "partition_mode": mode,
            "Xte": Xte, "yte": yte,
            "drift_map": drift_map,
            "drift_clients": drift_clients, "drift_schedule": drift_schedule,
            "fault_clients": (drift_clients if cfg.scenario == "fault" else set()),
            "dropped_clients": dropped_clients,
            "fdms_groups": fdms_groups,
            "concept_of": concept_of, "drift_maps": drift_maps_multi,
            "vshift_X": vshift_X, "vshift_Xte": vshift_Xte,
            "label_X": label_X, "label_y": label_y,
            "label_Xte": label_Xte, "label_yte": label_yte,
            "label_class_weight": label_class_weight,
            "label_calibration": label_calibration,
            "label_prior_calibration": label_prior_calibration,
            "client_label_hist": client_label_hist,
            "baseline_label_hist": baseline_label_hist,
            "label_target_hist": label_target_hist,
            "baseline_xmean": baseline_xmean, "baseline_xstd": baseline_xstd,
            "vshift_xmean": vshift_xmean, "vshift_xstd": vshift_xstd,
            "n_classes": n_classes, "input_shape": input_shape}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["drift", "fault", "dropout", "hetero", "staggered"], default="drift")
    ap.add_argument("--drift_mode", choices=["sudden", "gradual", "recurrent"], default="sudden")
    ap.add_argument("--drift_type", choices=["real", "virtual", "label"], default="real",
                    help="drift source (AC1.2.5): real P(y|x) / virtual P(x) / label P(y)")
    ap.add_argument("--virtual_shift", type=float, help="virtual drift input shift strength")
    ap.add_argument("--label_skew_frac", type=float, help="legacy label drift kept class fraction (unused by B1 long-tail)")
    ap.add_argument("--label_imbalance_factor", type=float, help="label drift long-tail imbalance factor")
    ap.add_argument("--label_min_per_class", type=int, help="minimum samples per present class after label resampling")
    ap.add_argument("--label_prior_strength", type=float, help="label-prior logit-adjustment strength")
    ap.add_argument("--fedlc_tau", type=float, help="FedLC logit calibration strength")
    ap.add_argument("--feddaa_T", type=int, help="FedDAA rounds between activation-relative rebuilds")
    ap.add_argument("--recur_gap", type=int, help="recurrent: rounds until drift back")
    ap.add_argument("--dropout_mode", choices=["correlated", "random", "fdms_clustered"], default="correlated")
    ap.add_argument("--dropout_rate", type=float, default=0.5, help="alpha for random dropout")
    ap.add_argument("--dropout_recover_round", type=int, default=0,
                    help=">0: clients return after this round (temporary outage window)")
    ap.add_argument("--fedprox_mu", type=float, help="override FedProx proximal strength")
    ap.add_argument("--byzantine_frac", type=float, help="fault: compromised client fraction")
    ap.add_argument("--byzantine_scale", type=float,
                    help="fault: scale for sign-flipped client updates")
    ap.add_argument("--robust_variant", choices=["distance_trimmed_mean", "coord_trimmed_mean"],
                    help="fault: robust aggregation variant")
    ap.add_argument("--hetero_beta", type=float, help="override hetero Dirichlet beta")
    ap.add_argument("--hetero_local_epochs", type=int, help="override hetero local epochs")
    ap.add_argument("--hetero_participation", type=float, help="override hetero participation")
    ap.add_argument("--llm", action="store_true", help="use real qwen3.8-max for LLMAgent (needs DASHSCOPE_API_KEY)")
    ap.add_argument("--decide_every", type=int, help="agent decides every K rounds (LLM cost control)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS, default="cifar10")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--drift_round", type=int)
    ap.add_argument("--drift_phase", choices=["early", "mid", "late"],
                    help="preset drift_round = {0.2,0.5,0.8} x rounds (reaction-speed sweep)")
    ap.add_argument("--participation", type=float,
                    help="<1.0: only this fraction of clients participate each round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no_plot", action="store_true", help="skip matplotlib recovery figure")
    ap.add_argument("--out_dir", help="directory for metrics, telemetry, and figures")
    ap.add_argument("--partition_mode", choices=PARTITION_MODES, default="noniid")
    ap.add_argument("--formal_evidence", action="store_true",
                    help="score only paper-supported actions as correct")
    ap.add_argument("--proxy_evidence", action="store_true",
                    help="B2 scoring: count supported and proxy evidence actions as correct")
    ap.add_argument("--evidence_report", action="store_true",
                    help="write results/evidence_report.md and exit")
    args = ap.parse_args()

    if args.evidence_report:
        print(f"saved {write_markdown()}")
        return

    cfg = Config()
    cfg.scenario = args.scenario
    cfg.drift_mode = args.drift_mode
    cfg.drift_type = args.drift_type
    cfg.dropout_mode = args.dropout_mode
    cfg.dropout_rate = args.dropout_rate
    cfg.dropout_recover_round = args.dropout_recover_round
    cfg.seed = args.seed
    cfg.partition_mode = args.partition_mode
    cfg.formal_evidence = args.formal_evidence
    cfg.proxy_evidence = args.proxy_evidence
    cfg.drift_fraction = DRIFT_FRACTION[args.scenario]
    cfg.dataset = args.dataset
    if cfg.scenario == "hetero":
        cfg.local_epochs = cfg.hetero_local_epochs   # more local drift for FedProx to fight
        cfg.participation = cfg.hetero_participation
    if args.synthetic: cfg.dataset = "synthetic"  # backward-compatible alias
    if args.smoke: cfg.smoke()
    if args.rounds: cfg.rounds = args.rounds
    if args.drift_phase:   # preset before explicit --drift_round so the latter wins
        cfg.drift_round = int({"early": 0.2, "mid": 0.5, "late": 0.8}[args.drift_phase] * cfg.rounds)
    if args.drift_round is not None: cfg.drift_round = args.drift_round
    if args.participation is not None: cfg.participation = args.participation
    if args.fedprox_mu is not None: cfg.fedprox_mu = args.fedprox_mu
    if args.byzantine_frac is not None: cfg.byzantine_frac = args.byzantine_frac
    if args.byzantine_scale is not None: cfg.byzantine_scale = args.byzantine_scale
    if args.robust_variant is not None: cfg.robust_variant = args.robust_variant
    if args.hetero_beta is not None: cfg.hetero_beta = args.hetero_beta
    if args.hetero_local_epochs is not None: cfg.local_epochs = args.hetero_local_epochs
    if args.hetero_participation is not None: cfg.participation = args.hetero_participation
    if args.virtual_shift is not None: cfg.virtual_shift = args.virtual_shift
    if args.label_skew_frac is not None: cfg.label_skew_frac = args.label_skew_frac
    if args.label_imbalance_factor is not None: cfg.label_imbalance_factor = args.label_imbalance_factor
    if args.label_min_per_class is not None: cfg.label_min_per_class = args.label_min_per_class
    if args.label_prior_strength is not None: cfg.label_prior_strength = args.label_prior_strength
    if args.fedlc_tau is not None: cfg.fedlc_tau = args.fedlc_tau
    if args.feddaa_T is not None:
        if args.feddaa_T < 1: ap.error("--feddaa_T must be >= 1")
        cfg.feddaa_T = args.feddaa_T
    if args.recur_gap is not None: cfg.recur_gap = args.recur_gap
    if args.decide_every: cfg.decide_every = args.decide_every
    if args.out_dir: cfg.out_dir = args.out_dir
    cfg.out_dir = str(partition_output_dir(cfg.out_dir, cfg.partition_mode))
    if not torch.cuda.is_available(): cfg.device = "cpu"
    os.makedirs(cfg.out_dir, exist_ok=True)

    set_seed(cfg.seed)
    data = build_data(cfg)
    eng = Engine(cfg, data)
    tag = output_tag(cfg)
    if args.llm:
        tag += "_qwen"
    # hetero = static extreme non-IID, no injection point -> event basis is r0
    event_round = 0 if cfg.scenario == "hetero" else cfg.drift_round
    if cfg.scenario == "dropout":
        if cfg.dropout_mode == "random":
            extra = f"dropout_mode=random alpha={cfg.dropout_rate}"
        elif cfg.dropout_mode == "fdms_clustered":
            extra = (f"dropout_mode=fdms_clustered alpha={cfg.dropout_rate} "
                     f"clusters={len(data.get('fdms_groups', []))}")
        else:
            extra = f"dropout_mode=correlated dropped={len(data['dropped_clients'])}/{cfg.num_clients}"
        if cfg.dropout_recover_round > 0:
            extra += f" (outage window [{cfg.drift_round},{cfg.dropout_recover_round}), then recover)"
    elif cfg.scenario == "hetero":
        extra = f"hetero_beta={cfg.hetero_beta} fedprox_mu={cfg.fedprox_mu} (static, no event)"
    elif cfg.scenario == "staggered":
        extra = f"n_concepts={cfg.n_concepts} stagger_gap={cfg.stagger_gap} (G1@{cfg.drift_round}, G2@{cfg.drift_round + cfg.stagger_gap})"
    else:
        extra = f"drift_clients={len(data['drift_clients'])}/{cfg.num_clients} mode={cfg.drift_mode}"
        if cfg.scenario == "fault":
            extra += f" byzantine_frac={cfg.byzantine_frac} robust_variant={cfg.robust_variant}"
        if cfg.drift_mode == "recurrent":
            extra += f" (away@{cfg.drift_round} back@{cfg.drift_round + cfg.recur_gap})"
    print(f"scenario={cfg.scenario} device={eng.device} dataset={cfg.dataset} "
          f"rounds={cfg.rounds} event@{event_round} {extra}\n")

    llm_query_fn = None
    if args.llm:
        from llm_backend import make_qwen_query_fn
        llm_query_fn = make_qwen_query_fn()
    agents = make_agents(cfg.scenario, cfg.seed, event_round, llm_query_fn, cfg.drift_type)
    histories = {}
    for ag in agents:
        set_seed(cfg.seed)
        histories[ag.name] = eng.run(ag, log=not args.quiet)
        if args.quiet:
            h = histories[ag.name]
            print(f"  done {ag.name:16s} final={h['acc'][-1]:.3f} tool={h['tool']}")
        else:
            print()

    for ag in agents:
        tlog = histories[ag.name]["telemetry"]
        if not tlog:
            continue
        with open(os.path.join(cfg.out_dir, f"telemetry_{tag}_{ag.name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tlog[0].keys()))
            w.writeheader()
            for row in tlog:
                w.writerow(row)

    rows = []
    for ag in agents:
        s = summarize(histories[ag.name], cfg, event_round)
        s["agent"] = ag.name
        rows.append(s)
    fields = ["agent", "final_acc", "post_drift_mean_acc", "post_event_worst_acc",
              "recovery_rounds", "tool", "tool_matched", "misdiagnosis_penalty",
              "formal_correct_tool", "evidence_status", "evidence_papers",
              "evidence_required_fix", "over_intervention", "diagnosis_accuracy",
              "switch_round", "switch_regret", "weight_entropy", "max_weight",
              "reweight_source", "robust_variant", "compromised_num", "robust_keep_n",
              "feddrift_cluster_purity", "feddrift_num_concepts",
              "feddrift_first_trigger_round", "feddrift_assignment_changed_clients"]
    with open(os.path.join(cfg.out_dir, f"metrics_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    if HAVE_MPL and not args.no_plot:
        part_beta = {"dropout": cfg.dropout_beta,
                     "hetero": cfg.hetero_beta}.get(cfg.scenario, cfg.dirichlet_beta)
        ylab = {"drift": "accuracy on NEW concept (follow drift)",
                "fault": "accuracy on original task (reject noise)",
                "dropout": "accuracy on original task (despite dropout)",
                "hetero": "accuracy on original task (extreme non-IID)",
                "staggered": "mean accuracy across concepts"}[cfg.scenario]
        plt.figure(figsize=(8, 4.8))
        for ag in agents:
            plt.plot(histories[ag.name]["acc"], label=ag.name, linewidth=1.8)
        if cfg.scenario != "hetero":            # hetero is static -> no injection line
            plt.axvline(cfg.drift_round, color="gray", ls=":", linewidth=1.2,
                        label=f"{cfg.scenario} @ r{cfg.drift_round}")
        plt.xlabel("round"); plt.ylabel(ylab)
        split = "IID" if cfg.partition_mode == "iid" else f"Dir({part_beta})"
        plt.title(f"DynFL-Bench B1 - scenario={tag} ({cfg.dataset}, {split})")
        plt.legend(loc="best", fontsize=9, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
        fig = os.path.join(cfg.out_dir, f"recovery_{tag}.png")
        plt.savefig(fig, dpi=130)
        print(f"saved {fig}")

    print("\n=== metrics ===")
    for r in rows:
        print(f"  {r['agent']:16s} final={r['final_acc']:.3f} post_mean={r['post_drift_mean_acc']} "
              f"worst={r['post_event_worst_acc']} tool={r['tool']}({r['tool_matched']}) "
              f"penalty={r['misdiagnosis_penalty']}")


if __name__ == "__main__":
    main()
