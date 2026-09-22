"""Supported-action validation runner.

Add new reproduction checks by registering one function in VALIDATORS.

Examples:
  python validate_supported.py all --synthetic --smoke --seeds 0 --require_supported
  python validate_supported.py fl_fdms --synthetic --smoke
  python validate_supported.py oort --seeds 0 44 56 --require_supported
  python validate_supported.py byzantine --seeds 0 44 56 --require_supported
  python validate_supported.py feddrift --seeds 0 44 56 --require_supported
  python validate_supported.py feddaa --drift_types real --seeds 0 44 56 --require_supported
  python validate_supported.py fl_fdms --seeds 0 44 56 --require_supported
"""
import argparse
import csv
import os
import statistics
import sys

import torch

from agents import (AutoSpawnConceptAgent, DiagnoseAgent, FriendSubstituteAgent,
                    HandleDropoutAgent, NoOpAgent, OracleAgent, RejectAgent,
                    SelectClientsAgent, SpawnConceptAgent, SwitchProxAgent)
from config import (Config, DRIFT_FRACTION, PARTITION_MODES, partition_output_dir)
from engine import Engine
from evidence import action_evidence_status
from run import build_data, set_seed
from scoring import summarize


def _mean(rows, agent, key):
    vals = [r[key] for r in rows if r["agent"] == agent and isinstance(r[key], (int, float))]
    return statistics.mean(vals) if vals else None


def _fdms_telemetry(hist):
    post = hist["telemetry"]
    substitutions = sum(int(r.get("friend_substitutions", 0) or 0) for r in post)
    sources = {r.get("fdms_similarity_source", "") for r in post}
    sims = [float(r.get("friend_similarity", 0) or 0) for r in post
            if float(r.get("friend_similarity", 0) or 0) > 0]
    errors = [float(r.get("fdms_substitution_error", 0) or 0) for r in post
              if isinstance(r.get("fdms_substitution_error"), (int, float))]
    candidates = [float(r.get("friend_candidate_count", 0) or 0) for r in post
                  if float(r.get("friend_candidate_count", 0) or 0) > 0]
    return {
        "fdms_substitutions": substitutions,
        "fdms_source_ok": bool({"update_cosine", "update_distance"} & sources),
        "fdms_friend_similarity_mean": round(statistics.mean(sims), 6) if sims else 0.0,
        "fdms_substitution_error_mean": round(statistics.mean(errors), 6) if errors else "n/a",
        "fdms_candidate_count_mean": round(statistics.mean(candidates), 6) if candidates else 0.0,
    }


def _oort_telemetry(hist):
    post = hist["telemetry"]
    next_selected = sum(1 for r in post if r.get("next_selected_clients"))
    used = sum(1 for r in post if r.get("selected_by_action"))
    fields_ok = all("oort_exploration" in r and "oort_round_threshold" in r
                    and "oort_prefer_duration" in r for r in post)
    return {
        "oort_fields_ok": fields_ok,
        "oort_next_selected_rounds": next_selected,
        "oort_selected_rounds": used,
        "oort_last_exploration": post[-1].get("oort_exploration", "") if post else "",
        "oort_last_round_threshold": post[-1].get("oort_round_threshold", "") if post else "",
    }


def _robust_telemetry(hist):
    post = hist["telemetry"]
    robust = [r for r in post if r.get("robust_variant") == "distance_trimmed_mean"]
    return {
        "robust_variant_ok": bool(robust),
        "robust_rounds": len(robust),
        "robust_max_compromised": max((int(r.get("compromised_num", 0) or 0) for r in robust), default=0),
        "robust_min_keep_n": min((int(r.get("robust_keep_n", 0) or 0) for r in robust), default=0),
        "robust_trigger_ok": any(r.get("trigger_source") == "update_norm_outlier" for r in robust),
    }


def _feddrift_telemetry(hist):
    post = hist["telemetry"]
    purities = [float(r["cluster_purity"]) for r in post
                if isinstance(r.get("cluster_purity"), (int, float))]
    sc_totals = [float(r.get("feddrift_sc_weight_total", 0) or 0) for r in post]
    sc_bins = [int(r.get("feddrift_sc_weight_bins", 0) or 0) for r in post]
    return {
        "feddrift_fields_ok": all("feddrift_assignment_source" in r
                                  and "feddrift_num_models" in r
                                  and "cluster_split_score" in r for r in post),
        "feddrift_auto_ok": any(r.get("feddrift_assignment_source") == "auto_hierarchical"
                                for r in post),
        "feddrift_source_ok": any(r.get("feddrift_source_variant") == "H_A_F_1_06_0"
                                  for r in post),
        "feddrift_sc_weights_ok": any(r.get("feddrift_train_schedule") == "compressed_sc_weights"
                                      and float(r.get("feddrift_sc_weight_total", 0) or 0) > 0
                                      for r in post),
        "feddrift_max_models": max((int(r.get("feddrift_num_models", 1) or 1)
                                    for r in post), default=1),
        "feddrift_model_cap_ok": all(int(r.get("feddrift_num_models", 1) or 1) <=
                                     int(r.get("feddrift_max_concepts", 4) or 4)
                                     for r in post),
        "feddrift_sc_weight_total_max": max(sc_totals, default=0.0),
        "feddrift_sc_weight_bins_max": max(sc_bins, default=0),
        "feddrift_drifted_rounds": sum(1 for r in post if r.get("feddrift_drifted_clients")),
        "feddrift_cluster_purity_mean": round(statistics.mean(purities), 6) if purities else "n/a",
    }


def _feddaa_telemetry(hist):
    post = hist["telemetry"]
    enabled = [r for r in post if r.get("feddaa_enabled")]
    return {
        "feddaa_fields_ok": all("feddaa_mode" in r and "feddaa_n_clusters" in r
                                and "feddaa_assignment_source" in r for r in post),
        "feddaa_enabled_rounds": len(enabled),
        "feddaa_modes": "|".join(sorted({r.get("feddaa_mode", "") for r in enabled if r.get("feddaa_mode")})),
        "feddaa_max_clusters": max((int(r.get("feddaa_n_clusters", 1) or 1)
                                    for r in post), default=1),
        "feddaa_source_ok": any(r.get("feddaa_assignment_source") == "prototype_silhouette_loss_label_weights"
                                and r.get("feddaa_cluster_source") == "prototype_silhouette"
                                for r in enabled),
        "feddaa_shift_detection_ok": any("feddaa_shift_clients" in r and "feddaa_clean_clients" in r
                                         for r in enabled),
        "feddaa_reweight_source_ok": any(r.get("reweight_source") == "feddaa_label_history_weights"
                                         for r in enabled),
        "feddaa_label_history_ok": any(r.get("feddaa_label_history") for r in enabled),
    }


def _cfg_for_fl_fdms(args, seed):
    cfg = Config()
    cfg.partition_mode = args.partition_mode
    cfg.scenario = "dropout"
    cfg.dropout_mode = "fdms_clustered"
    cfg.dropout_rate = args.dropout_rate
    cfg.seed = seed
    cfg.drift_fraction = DRIFT_FRACTION["dropout"]
    cfg.num_clients = 20
    cfg.fdms_clusters = 5
    cfg.samples_per_client = 1000
    cfg.batch_size = 200
    cfg.local_epochs = 5
    cfg.lr0 = 0.1
    cfg.drift_round = 1
    if args.synthetic:
        cfg.dataset = "synthetic"
    if args.smoke:
        cfg.smoke()
        cfg.dropout_mode = "fdms_clustered"  # smoke() should not change protocol.
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.drift_round is not None:
        cfg.drift_round = args.drift_round
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def _cfg_for_feddaa(args, seed, drift_type):
    cfg = Config()
    cfg.partition_mode = args.partition_mode
    cfg.scenario = "drift"
    cfg.drift_type = drift_type
    cfg.seed = seed
    cfg.formal_evidence = True
    cfg.drift_fraction = DRIFT_FRACTION["drift"]
    if args.synthetic:
        cfg.dataset = "synthetic"
    if args.smoke:
        cfg.smoke()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.drift_round is not None:
        cfg.drift_round = args.drift_round
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def _cfg_for_byzantine(args, seed):
    cfg = Config()
    cfg.partition_mode = args.partition_mode
    cfg.scenario = "fault"
    cfg.seed = seed
    cfg.formal_evidence = True
    cfg.drift_fraction = DRIFT_FRACTION["fault"]
    cfg.byzantine_frac = args.byzantine_frac
    cfg.robust_variant = args.robust_variant
    if args.synthetic:
        cfg.dataset = "synthetic"
    if args.smoke:
        cfg.smoke()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.drift_round is not None:
        cfg.drift_round = args.drift_round
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def _cfg_for_oort(args, seed):
    cfg = Config()
    cfg.partition_mode = args.partition_mode
    cfg.scenario = "hetero"
    cfg.seed = seed
    cfg.formal_evidence = True
    cfg.drift_fraction = DRIFT_FRACTION["hetero"]
    cfg.local_epochs = args.hetero_local_epochs or cfg.hetero_local_epochs
    cfg.participation = args.hetero_participation or cfg.hetero_participation
    if args.hetero_beta is not None:
        cfg.hetero_beta = args.hetero_beta
    if args.synthetic:
        cfg.dataset = "synthetic"
    if args.smoke:
        cfg.smoke()
        cfg.local_epochs = args.hetero_local_epochs or cfg.hetero_local_epochs
        cfg.participation = args.hetero_participation or cfg.hetero_participation
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def _cfg_for_feddrift(args, seed):
    cfg = Config()
    cfg.partition_mode = args.partition_mode
    cfg.scenario = "staggered"
    cfg.seed = seed
    cfg.formal_evidence = True
    cfg.drift_fraction = DRIFT_FRACTION["staggered"]
    if args.synthetic:
        cfg.dataset = "synthetic"
    if args.smoke:
        cfg.smoke()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.drift_round is not None:
        cfg.drift_round = args.drift_round
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


def validate_fl_fdms(args):
    rows = []
    for seed in args.seeds:
        cfg = _cfg_for_fl_fdms(args, seed)
        event_round = cfg.drift_round
        full_cfg = _cfg_for_fl_fdms(args, seed)
        full_cfg.dropout_rate = 0.0
        set_seed(seed)
        data = build_data(full_cfg)
        eng = Engine(full_cfg, data)
        agent = NoOpAgent()
        set_seed(seed)
        hist = eng.run(agent, log=False)
        row = summarize(hist, full_cfg, event_round)
        row.update({"target": "fl_fdms", "seed": seed, "agent": "fl_full",
                    "fdms_substitutions": 0, "fdms_source_ok": False,
                    "fdms_friend_similarity_mean": 0.0,
                    "fdms_substitution_error_mean": "n/a",
                    "fdms_candidate_count_mean": 0.0})
        rows.append(row)

        roster = [(NoOpAgent(), "fl_dropout", 0),
                  (HandleDropoutAgent(), "fl_stale", 1),
                  (FriendSubstituteAgent(), "fl_fdms", 1)]
        for agent, name, warmup in roster:
            cfg.fdms_warmup_rounds = warmup
            set_seed(seed)
            data = build_data(cfg)
            eng = Engine(cfg, data)
            set_seed(seed)
            hist = eng.run(agent, log=False)
            row = summarize(hist, cfg, event_round)
            row.update({"target": "fl_fdms", "seed": seed, "agent": name})
            if name == "fl_fdms":
                row.update(_fdms_telemetry(hist))
            else:
                row.update({"fdms_substitutions": 0, "fdms_source_ok": False,
                            "fdms_friend_similarity_mean": 0.0,
                            "fdms_substitution_error_mean": "n/a",
                            "fdms_candidate_count_mean": 0.0})
            rows.append(row)

    full = _mean(rows, "fl_full", "post_drift_mean_acc")
    noop = _mean(rows, "fl_dropout", "post_drift_mean_acc")
    stale = _mean(rows, "fl_stale", "post_drift_mean_acc")
    fdms = _mean(rows, "fl_fdms", "post_drift_mean_acc")
    fdms_rows = [r for r in rows if r["agent"] == "fl_fdms"]
    mechanism_ok = all(r["fdms_source_ok"] and r["fdms_substitutions"] > 0
                       and r["fdms_candidate_count_mean"] > 0
                       for r in fdms_rows)
    performance_ok = (fdms is not None and noop is not None and stale is not None
                      and fdms > noop and fdms >= stale)
    return rows, {
        "target": "fl_fdms",
        "mechanism_ok": mechanism_ok,
        "performance_ok": performance_ok,
        "ready_for_supported": mechanism_ok and performance_ok,
        "full_mean": full,
        "noop_mean": noop,
        "stale_mean": stale,
        "fdms_mean": fdms,
    }


def validate_feddaa(args):
    rows = []
    drift_types = args.drift_types
    for seed in args.seeds:
        for drift_type in drift_types:
            cfg = _cfg_for_feddaa(args, seed, drift_type)
            event_round = cfg.drift_round
            set_seed(seed)
            data = build_data(cfg)
            eng = Engine(cfg, data)
            for agent in (NoOpAgent(), DiagnoseAgent(),
                          OracleAgent("drift", event_round, drift_type)):
                set_seed(seed)
                hist = eng.run(agent, log=False)
                row = summarize(hist, cfg, event_round)
                row.update({"target": "feddaa", "seed": seed, "agent": agent.name,
                            "drift_type": drift_type})
                if agent.name in ("diagnose", "oracle"):
                    row.update(_feddaa_telemetry(hist))
                else:
                    row.update({"feddaa_fields_ok": False, "feddaa_enabled_rounds": 0,
                                "feddaa_modes": "", "feddaa_max_clusters": 1,
                                "feddaa_source_ok": False,
                                "feddaa_shift_detection_ok": False,
                                "feddaa_reweight_source_ok": False,
                                "feddaa_label_history_ok": False})
                rows.append(row)

    expected = {"real": "drift_adapt", "virtual": "moment_align_adapt", "label": "reweight"}
    subject_rows = [r for r in rows if r["agent"] == "oracle"]
    mechanism_ok = all(r["tool"] == expected[r["drift_type"]]
                       and r["tool_matched"] == "yes"
                       and r["evidence_status"] == action_evidence_status("drift_adapt")
                       and r["evidence_papers"] == "feddaa"
                       and r["feddaa_fields_ok"]
                       and r["feddaa_source_ok"]
                       and r["feddaa_shift_detection_ok"]
                       and r["feddaa_enabled_rounds"] > 0
                       and r["feddaa_modes"] == r["drift_type"]
                       and (r["drift_type"] != "label" or
                            (r["feddaa_reweight_source_ok"] and r["feddaa_label_history_ok"]))
                       for r in subject_rows)
    diagnose_rows = [r for r in rows if r["agent"] == "diagnose"]
    diagnosis_ok = all(r["tool_matched"] == "yes" for r in diagnose_rows)
    performance_ok = all(
        (_mean([r for r in rows if r["drift_type"] == dt], "oracle", "post_drift_mean_acc") or 0) >=
        (_mean([r for r in rows if r["drift_type"] == dt], "noop", "post_drift_mean_acc") or 0)
        for dt in drift_types
    )
    return rows, {
        "target": "feddaa",
        "mechanism_ok": mechanism_ok,
        "diagnosis_ok": diagnosis_ok,
        "performance_ok": performance_ok,
        "ready_for_supported": mechanism_ok and diagnosis_ok and performance_ok,
        "drift_types": "|".join(drift_types),
        "noop_mean": _mean(rows, "noop", "post_drift_mean_acc"),
        "diagnose_mean": _mean(rows, "diagnose", "post_drift_mean_acc"),
        "oracle_mean": _mean(rows, "oracle", "post_drift_mean_acc"),
    }


def validate_feddrift(args):
    rows = []
    for seed in args.seeds:
        cfg = _cfg_for_feddrift(args, seed)
        event_round = cfg.drift_round
        set_seed(seed)
        data = build_data(cfg)
        eng = Engine(cfg, data)
        for agent in (NoOpAgent(), SpawnConceptAgent(), AutoSpawnConceptAgent(),
                      OracleAgent("staggered", event_round)):
            set_seed(seed)
            hist = eng.run(agent, log=False)
            row = summarize(hist, cfg, event_round)
            row.update({"target": "feddrift", "seed": seed, "agent": agent.name})
            if agent.name == "spawn_concept_auto":
                row.update(_feddrift_telemetry(hist))
            else:
                row.update({"feddrift_fields_ok": False, "feddrift_auto_ok": False,
                            "feddrift_max_models": 1, "feddrift_drifted_rounds": 0,
                            "feddrift_cluster_purity_mean": "n/a"})
            rows.append(row)

    noop = _mean(rows, "noop", "post_drift_mean_acc")
    truth = _mean(rows, "spawn_concept", "post_drift_mean_acc")
    auto = _mean(rows, "spawn_concept_auto", "post_drift_mean_acc")
    oracle = _mean(rows, "oracle", "post_drift_mean_acc")
    auto_rows = [r for r in rows if r["agent"] == "spawn_concept_auto"]
    mechanism_ok = all(r["tool_matched"] == "yes"
                       and r["evidence_status"] == action_evidence_status("spawn_concept_auto")
                       and r["evidence_papers"] == "feddrift"
                       and r["feddrift_fields_ok"]
                       and r["feddrift_auto_ok"]
                       and r["feddrift_source_ok"]
                       and r["feddrift_sc_weights_ok"]
                       and r["feddrift_model_cap_ok"]
                       and r["feddrift_max_models"] > 1
                       for r in auto_rows)
    performance_ok = auto is not None and noop is not None and auto > noop
    return rows, {
        "target": "feddrift",
        "mechanism_ok": mechanism_ok,
        "performance_ok": performance_ok,
        "ready_for_supported": mechanism_ok and performance_ok,
        "noop_mean": noop,
        "spawn_concept_oracle_proxy_mean": truth,
        "spawn_concept_auto_mean": auto,
        "oracle_mean": oracle,
    }


def validate_byzantine(args):
    rows = []
    for seed in args.seeds:
        cfg = _cfg_for_byzantine(args, seed)
        set_seed(seed)
        data = build_data(cfg)
        eng = Engine(cfg, data)
        for agent in (NoOpAgent(), RejectAgent(), OracleAgent("fault", cfg.drift_round)):
            set_seed(seed)
            hist = eng.run(agent, log=False)
            row = summarize(hist, cfg, cfg.drift_round)
            row.update({"target": "byzantine", "seed": seed, "agent": agent.name})
            if agent.name in ("reject_robust", "oracle"):
                row.update(_robust_telemetry(hist))
            else:
                row.update({"robust_variant_ok": False, "robust_rounds": 0,
                            "robust_max_compromised": 0, "robust_min_keep_n": 0,
                            "robust_trigger_ok": False})
            rows.append(row)

    noop = _mean(rows, "noop", "post_drift_mean_acc")
    reject = _mean(rows, "reject_robust", "post_drift_mean_acc")
    oracle = _mean(rows, "oracle", "post_drift_mean_acc")
    robust_rows = [r for r in rows if r["agent"] == "oracle"]
    mechanism_ok = all(r["tool_matched"] == "yes"
                       and r["evidence_status"] == action_evidence_status("robust")
                       and r["evidence_papers"] == "byz_trimmed_mean"
                       and r["robust_variant_ok"]
                       and r["robust_max_compromised"] > 0
                       and r["robust_min_keep_n"] > 0
                       for r in robust_rows)
    performance_ok = oracle is not None and noop is not None and oracle >= noop
    return rows, {
        "target": "byzantine",
        "mechanism_ok": mechanism_ok,
        "performance_ok": performance_ok,
        "ready_for_supported": mechanism_ok and performance_ok,
        "noop_mean": noop,
        "reject_robust_mean": reject,
        "oracle_mean": oracle,
        "reject_triggered": any(r["agent"] == "reject_robust" and r["tool"] == "robust" for r in rows),
    }


def validate_oort(args):
    rows = []
    for seed in args.seeds:
        cfg = _cfg_for_oort(args, seed)
        set_seed(seed)
        data = build_data(cfg)
        eng = Engine(cfg, data)
        for agent in (NoOpAgent(), SwitchProxAgent(), SelectClientsAgent()):
            set_seed(seed)
            hist = eng.run(agent, log=False)
            row = summarize(hist, cfg, 0)
            row.update({"target": "oort", "seed": seed, "agent": agent.name})
            if agent.name == "select_clients":
                row.update(_oort_telemetry(hist))
            else:
                row.update({"oort_fields_ok": False, "oort_next_selected_rounds": 0,
                            "oort_selected_rounds": 0, "oort_last_exploration": "",
                            "oort_last_round_threshold": ""})
            rows.append(row)

    noop = _mean(rows, "noop", "post_drift_mean_acc")
    prox = _mean(rows, "switch_prox", "post_drift_mean_acc")
    oort = _mean(rows, "select_clients", "post_drift_mean_acc")
    oort_rows = [r for r in rows if r["agent"] == "select_clients"]
    mechanism_ok = all(r["tool_matched"] == "yes"
                       and r["evidence_status"] == action_evidence_status("select_clients")
                       and r["evidence_papers"] == "oort"
                       and r["oort_fields_ok"]
                       and r["oort_next_selected_rounds"] > 0
                       for r in oort_rows)
    performance_ok = oort is not None and noop is not None and oort > noop
    return rows, {
        "target": "oort",
        "mechanism_ok": mechanism_ok,
        "performance_ok": performance_ok,
        "ready_for_supported": mechanism_ok and performance_ok,
        "noop_mean": noop,
        "switch_prox_mean": prox,
        "oort_mean": oort,
    }


VALIDATORS = {"byzantine": validate_byzantine, "feddaa": validate_feddaa,
              "feddrift": validate_feddrift,
              "fl_fdms": validate_fl_fdms,
              "oort": validate_oort}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["all"] + sorted(VALIDATORS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 44, 56])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--drift_round", type=int)
    ap.add_argument("--drift_types", nargs="+", choices=["real"], default=["real"],
                    help="FedDAA validator covers real drift only")
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--byzantine_frac", type=float, default=0.2)
    ap.add_argument("--robust_variant", choices=["distance_trimmed_mean", "coord_trimmed_mean"],
                    default="distance_trimmed_mean")
    ap.add_argument("--hetero_beta", type=float)
    ap.add_argument("--hetero_local_epochs", type=int)
    ap.add_argument("--hetero_participation", type=float)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--partition_mode", choices=PARTITION_MODES, default="noniid")
    ap.add_argument("--require_supported", action="store_true",
                    help="exit nonzero unless the target is ready to mark supported")
    args = ap.parse_args()

    args.out_dir = str(partition_output_dir(args.out_dir, args.partition_mode))
    os.makedirs(args.out_dir, exist_ok=True)
    targets = sorted(VALIDATORS) if args.target == "all" else [args.target]
    summaries = []
    for target in targets:
        print(f"\n===== validating {target} =====")
        rows, summary = VALIDATORS[target](args)
        workpoint_role = ("iid_system_control" if target == "oort" and
                          args.partition_mode == "iid" else "benchmark")
        for row in rows:
            row["partition_mode"] = args.partition_mode
            row["workpoint_role"] = workpoint_role
        summary["partition_mode"] = args.partition_mode
        summary["workpoint_role"] = workpoint_role
        path = os.path.join(args.out_dir, f"validation_{target}.csv")
        fields = sorted({k for row in rows for k in row})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})
        summaries.append(summary)
        print(f"saved {path}")
        for k, v in summary.items():
            print(f"{k}: {v}")

    if args.target == "all":
        path = os.path.join(args.out_dir, "validation_all.csv")
        fields = sorted({k for row in summaries for k in row})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in summaries:
                w.writerow({k: row.get(k, "") for k in fields})
        print(f"\nsaved {path}")

    if args.require_supported and not all(s["ready_for_supported"] for s in summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
