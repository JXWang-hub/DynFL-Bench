"""Scoring. Metrics are computed on the 对症 accuracy curve produced by engine.

Correct tools come from evidence.py. Default mode keeps legacy experiments
comparable; formal_evidence=True accepts non-proxy compliant actions;
proxy_evidence=True also accepts proxy actions for B2 stage runs.
"""
import math

from evidence import PAPERS, correct_actions, evidence_for


LEGACY_CORRECT_TOOL = {"drift": "lr_reset", "fault": "robust", "dropout": "dropout_handle",
                       "hetero": "fedprox", "staggered": "spawn_concept"}


def evidence_subtype(cfg):
    if cfg.scenario != "drift":
        return None
    if cfg.drift_mode == "recurrent":
        return "recurrent"
    return cfg.drift_type


def correct_tools_for(cfg):
    subtype = evidence_subtype(cfg)
    formal = getattr(cfg, "formal_evidence", False)
    proxy = getattr(cfg, "proxy_evidence", False)
    tools = correct_actions(cfg.scenario, subtype=subtype, formal=formal, proxy=proxy)
    if cfg.scenario == "hetero":
        tools = tuple(t for t in tools if t == "select_clients")
        if getattr(cfg, "participation", 1.0) >= 1.0:
            tools = ()
    if tools or formal or proxy:
        return tools
    # ponytail: compatibility path for legacy scenarios only.
    if cfg.scenario == "drift" and cfg.drift_type == "label":
        return ("label_prior_adapt",)
    if cfg.scenario == "drift" and cfg.drift_type == "virtual":
        return ("moment_align_adapt",)
    if cfg.scenario == "drift" and cfg.drift_type == "real":
        return ("drift_adapt",)
    return (LEGACY_CORRECT_TOOL.get(cfg.scenario),)


def evidence_summary(cfg, tool):
    subtype = evidence_subtype(cfg)
    entries = evidence_for(cfg.scenario, tool, subtype=subtype)
    if not entries and tool == "none":
        entries = evidence_for("stable", "no_op")
    if not entries:
        return "missing", "", "no evidence entry"
    best = entries[0]
    papers = ";".join(PAPERS[p].key for p in best.papers)
    return best.status, papers, best.required_fix


def misdiagnosis_penalty(tool, correct_tools):
    """F4.6: penalize tool/disturbance mismatch. Using the WRONG active tool
    (e.g. robust on a genuine drift -> rejects legitimate clients) is the heavy
    penalty; failing to act at all (none) is a lighter miss. All three B1
    scenarios require action, so 'none' always scores the miss penalty here."""
    if tool in correct_tools:
        return 0.0
    if tool == "none":
        return 0.5      # missed: no diagnosis acted on, but no active harm
    return 1.0          # wrong tool family applied (heavy)


def action_sequence_score(hist, correct_tools):
    """Score every distinct intervention that became active, not only the first."""
    actions = []
    for row in hist.get("telemetry", []):
        bundle = row.get("action_bundle", row.get("action", "no_op"))
        for action in bundle.split("|"):
            if row.get("decision_due", True) and action != "no_op" and action not in actions:
                actions.append(action)
    if not actions:
        return actions, 0.5
    return actions, max(misdiagnosis_penalty(action, correct_tools) for action in actions)


def recovery_time(hist, drift_round):
    acc = hist["acc"]
    if drift_round >= len(acc):
        return None
    pre = max(acc[:drift_round]) if drift_round > 0 else acc[0]
    dipped = any(a < pre - 0.01 for a in acc[drift_round:])
    if not dipped:
        return 0
    for i in range(drift_round, len(acc)):
        if acc[i] >= pre:
            return i - drift_round
    return None


def post_drift_stats(hist, drift_round):
    post = hist["acc"][drift_round:]
    if not post:
        return None, None
    return sum(post) / len(post), min(post)


def diagnosis_accuracy(hist, event_round, true_type):
    """F4.4 / FedDAA: fraction of post-event rounds whose sub-type diagnosis matches
    the ground truth. n/a if no sub-type (non-drift) or the agent never diagnosed."""
    if not true_type:
        return "n/a"
    post = [t.get("diagnosis", "") for t in hist["telemetry"][event_round:]]
    post = [d for d in post if d]
    if not post:
        return "n/a"
    return round(sum(1 for d in post if d == true_type) / len(post), 2)


def summarize(hist, cfg, event_round):
    """event_round = the post-event window basis AND the oracle switch round.
    drift/fault/dropout pass cfg.drift_round; hetero passes 0 (static non-IID:
    high from r0, so post window = whole run and the oracle acts at r0)."""
    mean_post, min_post = post_drift_stats(hist, event_round)
    rt = recovery_time(hist, event_round)
    sw = hist["switch_round"]
    correct_tools = correct_tools_for(cfg)
    action_sequence, sequence_penalty = action_sequence_score(hist, correct_tools)
    matched = "n/a" if not correct_tools else ("yes" if sequence_penalty == 0 else "no")
    penalty = "n/a" if not correct_tools else sequence_penalty
    ev_status, ev_papers, ev_fix = evidence_summary(cfg, hist["tool"])
    trigger_target = event_round
    transition_mean = "n/a"
    post_full_mean = mean_post if mean_post is not None else "n/a"
    if cfg.scenario == "drift" and cfg.drift_mode == "gradual":
        half_wave = max(0, math.ceil(0.5 / max(float(cfg.drift_wave_frac), 1e-9)) - 1)
        final_wave = max(0, math.ceil(1.0 / max(float(cfg.drift_wave_frac), 1e-9)) - 1)
        trigger_target = event_round + half_wave * int(cfg.drift_wave_every)
        transition_end = min(len(hist["acc"]), event_round + final_wave * int(cfg.drift_wave_every))
        transition = hist["acc"][event_round:transition_end]
        full = hist["acc"][transition_end:]
        transition_mean = round(sum(transition) / len(transition), 4) if transition else "n/a"
        post_full_mean = round(sum(full) / len(full), 4) if full else "n/a"
    regret = (max(event_round - sw, sw - trigger_target, 0)
              if sw is not None else "n/a")
    worst_seq = hist.get("worst_acc", [])
    worst_post = worst_seq[event_round:]
    pew = round(sum(worst_post) / len(worst_post), 4) if worst_post else "n/a"
    # F4.5 over-intervention: non-no_op actions taken BEFORE the disturbance (should be no_op)
    over = sum(1 for (r, a) in hist["actions"] if r < event_round and a != "no_op")
    diag_acc = diagnosis_accuracy(hist, event_round,
                                  cfg.drift_type if cfg.scenario == "drift" else None)
    weight_entropy_seq = hist.get("weight_entropy", [])
    max_weight_seq = hist.get("max_weight", [])
    reweight_source_seq = hist.get("reweight_source", [])
    weight_entropy = round(sum(weight_entropy_seq[event_round:]) / len(weight_entropy_seq[event_round:]), 6) if weight_entropy_seq[event_round:] else "n/a"
    max_weight = round(max(max_weight_seq[event_round:]), 6) if max_weight_seq[event_round:] else "n/a"
    reweight_source = reweight_source_seq[-1] if reweight_source_seq else "n/a"
    robust_variants = [v for v in hist.get("robust_variant", [])[event_round:] if v != "none"]
    compromised = hist.get("compromised_num", [])[event_round:]
    keep_n = hist.get("robust_keep_n", [])[event_round:]
    fed_rows = hist.get("telemetry", [])[event_round:] if cfg.scenario == "staggered" else []
    purities = [float(row["feddrift_cluster_purity"]) for row in fed_rows
                if row.get("feddrift_cluster_purity") not in (None, "", "n/a")]
    concepts = [int(row["feddrift_num_concepts"]) for row in fed_rows
                if row.get("feddrift_num_concepts") not in (None, "", "n/a")]
    triggers = [int(row["feddrift_first_trigger_round"]) for row in fed_rows
                if row.get("feddrift_first_trigger_round") not in (None, "", "n/a")]
    assignment_changes = [int(row.get("feddrift_assignment_changed_clients", 0) or 0)
                          for row in fed_rows]
    aris = [float(row["feddrift_ari"]) for row in fed_rows
            if row.get("feddrift_ari") not in (None, "", "n/a")]
    nmis = [float(row["feddrift_nmi"]) for row in fed_rows
            if row.get("feddrift_nmi") not in (None, "", "n/a")]
    learner_errors = [int(row["feddrift_learner_count_error"]) for row in fed_rows
                      if row.get("feddrift_learner_count_error") not in (None, "", "n/a")]
    split_counts = [int(row.get("feddrift_split_count", 0)) for row in fed_rows]
    merge_counts = [int(row.get("feddrift_merge_count", 0)) for row in fed_rows]
    reuse_counts = [int(row.get("feddrift_reused_clients", 0)) for row in fed_rows]
    assignment_records = [int(row.get("feddrift_assignment_records", 0)) for row in fed_rows]
    daa_rows = [row for row in hist.get("telemetry", []) if row.get("feddaa_enabled")]
    daa_clusters = [int(row.get("feddaa_n_clusters", 1)) for row in daa_rows]
    daa_rebuilds = [int(row.get("feddaa_rebuild_count", 0)) for row in daa_rows]
    daa_churn = [int(row.get("feddaa_assignment_changed_clients", 0)) for row in daa_rows]
    daa_bytes = [int(row.get("feddaa_report_bytes", 0)) for row in daa_rows]
    daa_evaluations = [int(row.get("feddaa_model_evaluations", 0)) for row in daa_rows]
    llm_rows = [row for row in hist.get("telemetry", []) if row.get("llm_called")]
    llm_latencies = [float(row.get("llm_latency_ms", 0.0)) for row in llm_rows]
    llm_tokens = sum(int(row.get("llm_prompt_tokens", 0)) +
                     int(row.get("llm_completion_tokens", 0)) for row in llm_rows)
    post_rows = hist.get("telemetry", [])[event_round:]
    selected_rate = (sum(bool(row.get("selected_by_action")) for row in post_rows) / len(post_rows)
                     if post_rows else "n/a")
    switch_rows = [row for row in hist.get("telemetry", []) if row.get("round") == sw]
    switch_ratio = switch_rows[0].get("update_norm_ratio", "n/a") if switch_rows else "n/a"
    return {
        "final_acc": round(hist["acc"][-1], 4),
        "post_drift_mean_acc": round(mean_post, 4) if mean_post is not None else "n/a",
        "post_event_worst_acc": pew,
        "transition_mean_acc": transition_mean,
        "post_full_drift_mean_acc": post_full_mean,
        "recovery_rounds": rt if rt is not None else "n/a",
        "tool": hist["tool"],
        "action_sequence": "|".join(action_sequence) if action_sequence else "none",
        "active_action_count": len(action_sequence),
        "wrong_action_count": sum(action not in correct_tools for action in action_sequence),
        "tool_matched": matched,
        "misdiagnosis_penalty": penalty,
        "formal_correct_tool": "|".join(correct_tools) if correct_tools else "n/a",
        "evidence_status": ev_status,
        "evidence_papers": ev_papers,
        "evidence_required_fix": ev_fix,
        "over_intervention": over,
        "diagnosis_accuracy": diag_acc,
        "switch_round": sw if sw is not None else "n/a",
        "trigger_target_round": trigger_target,
        "switch_regret": regret,
        "switch_update_norm_ratio": switch_ratio,
        "selected_action_rate": round(selected_rate, 4) if isinstance(selected_rate, float) else selected_rate,
        "weight_entropy": weight_entropy,
        "max_weight": max_weight,
        "reweight_source": reweight_source,
        "robust_variant": robust_variants[-1] if robust_variants else "n/a",
        "compromised_num": max(compromised) if compromised else "n/a",
        "robust_keep_n": max(keep_n) if keep_n else "n/a",
        "feddrift_cluster_purity": round(purities[-1], 4) if purities else "n/a",
        "feddrift_num_concepts": concepts[-1] if concepts else "n/a",
        "feddrift_first_trigger_round": min(triggers) if triggers else "n/a",
        "feddrift_assignment_changed_clients": sum(assignment_changes) if assignment_changes else "n/a",
        "feddrift_ari": round(aris[-1], 4) if aris else "n/a",
        "feddrift_nmi": round(nmis[-1], 4) if nmis else "n/a",
        "feddrift_learner_count_error": learner_errors[-1] if learner_errors else "n/a",
        "feddrift_false_split_peak": max(learner_errors) if learner_errors else "n/a",
        "feddrift_split_count": max(split_counts) if split_counts else 0,
        "feddrift_merge_count": sum(merge_counts) if merge_counts else 0,
        "feddrift_reused_clients": sum(reuse_counts) if reuse_counts else 0,
        "feddrift_assignment_records": assignment_records[-1] if assignment_records else 0,
        "feddaa_n_clusters": daa_clusters[-1] if daa_clusters else "n/a",
        "feddaa_rebuild_count": daa_rebuilds[-1] if daa_rebuilds else 0,
        "feddaa_assignment_changed_clients": sum(daa_churn),
        "feddaa_report_bytes": sum(daa_bytes),
        "feddaa_model_evaluations": sum(daa_evaluations),
        "llm_calls": len(llm_rows),
        "llm_total_tokens": llm_tokens,
        "llm_cost_usd": round(sum(float(row.get("llm_cost_usd", 0.0)) for row in llm_rows), 6),
        "llm_latency_ms_mean": round(sum(llm_latencies) / len(llm_latencies), 3) if llm_latencies else "n/a",
        "llm_latency_ms_p95": round(sorted(llm_latencies)[max(0, math.ceil(0.95 * len(llm_latencies)) - 1)], 3) if llm_latencies else "n/a",
        "decision_trace": [
            {key: row.get(key, "") for key in (
                "round", "action", "reasoning", "diagnosis", "llm_called", "llm_model",
                "llm_latency_ms", "llm_prompt_tokens", "llm_completion_tokens", "llm_cost_usd"
            )}
            for row in hist.get("telemetry", []) if row.get("decision_due", True)
        ],
    }
