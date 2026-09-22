"""Telemetry layer (B1): the agent's only window into training.

Surfaces symptoms (accuracy trend, loss variance, update divergence, current LR,
participation rate) but NEVER the ground-truth disturbance type/round (AC2.2.2).
The agent must DIAGNOSE drift vs fault vs dropout from these symptoms.
"""
import math
import numpy as np
from observation_contract import (
    PUBLIC_OBSERVATION_SCHEMA, public_observation, serialize_public_observation,
)


def norm_stats(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    median = float(np.median(values)) if values.size else 0.0
    return {
        "mean": round(float(values.mean()), 4) if values.size else 0.0,
        "var": round(float(values.var()), 4) if values.size else 0.0,
        "max": round(maximum, 4),
        "median": round(median, 4),
        "ratio": round(maximum / max(median, 1e-9), 2) if values.size else 0.0,
    }


def clustering_metrics(assignment, truth):
    """Permutation-invariant small-cluster metrics; no sklearn dependency."""
    clients = sorted(set(assignment) & set(truth))
    if not clients:
        return {"purity": "n/a", "ari": "n/a", "nmi": "n/a",
                "learner_count_error": "n/a"}
    predicted = {label: [ci for ci in clients if assignment[ci] == label]
                 for label in set(assignment[ci] for ci in clients)}
    actual = {label: [ci for ci in clients if truth[ci] == label]
              for label in set(truth[ci] for ci in clients)}
    contingency = [[len(set(rows) & set(cols)) for cols in actual.values()]
                   for rows in predicted.values()]
    n = len(clients)
    row_sums = [sum(row) for row in contingency]
    col_sums = [sum(contingency[i][j] for i in range(len(contingency)))
                for j in range(len(actual))]
    purity = sum(max(row, default=0) for row in contingency) / n
    choose2 = lambda value: value * (value - 1) / 2
    pairs = choose2(n)
    index = sum(choose2(value) for row in contingency for value in row)
    row_pairs = sum(choose2(value) for value in row_sums)
    col_pairs = sum(choose2(value) for value in col_sums)
    expected = row_pairs * col_pairs / pairs if pairs else 0.0
    maximum = 0.5 * (row_pairs + col_pairs)
    ari = ((index - expected) / (maximum - expected)
           if maximum > expected else 1.0)
    mutual = sum((value / n) * math.log((value * n) / (row_sums[i] * col_sums[j]))
                 for i, row in enumerate(contingency)
                 for j, value in enumerate(row) if value)
    h_pred = -sum((value / n) * math.log(value / n) for value in row_sums if value)
    h_true = -sum((value / n) * math.log(value / n) for value in col_sums if value)
    nmi = mutual / math.sqrt(h_pred * h_true) if h_pred and h_true else 1.0 if h_pred == h_true else 0.0
    return {
        "purity": round(float(purity), 6),
        "ari": round(float(ari), 6),
        "nmi": round(float(nmi), 6),
        "learner_count_error": abs(len(predicted) - len(actual)),
    }


def collect(round_idx, acc, acc_history, client_losses, update_norms,
            current_lr, participation_rate, use_robust, total_rounds,
            input_shift=0.0, label_shift=0.0):
    cl = np.asarray(client_losses, dtype=np.float64)
    un = norm_stats(update_norms)
    if len(acc_history) >= 4:
        recent_delta = acc_history[-1] - max(acc_history[-4:-1])
    else:
        recent_delta = 0.0
    return {
        "round": round_idx,
        "total_rounds": total_rounds,
        "global_acc": round(float(acc), 4),
        "acc_delta_3": round(float(recent_delta), 4),
        "client_loss_mean": round(float(cl.mean()), 4) if cl.size else 0.0,
        "client_loss_var": round(float(cl.var()), 4) if cl.size else 0.0,
        "update_norm_mean": un["mean"],
        "update_norm_var": un["var"],
        "update_norm_max": un["max"],
        "update_norm_median": un["median"],
        "update_norm_ratio": un["ratio"],
        "input_shift": round(float(input_shift), 3),      # virtual drift: P(x) moved
        "label_shift": round(float(label_shift), 3),      # roster-matched aggregate P(y) shift
        "current_lr": round(float(current_lr), 5),
        "participation_rate": round(float(participation_rate), 3),
        "planned_participation_rate": round(float(participation_rate), 3),
        "availability_rate": round(float(participation_rate), 3),
        "participation_gap": 0.0,
        "active_actions": [],
        "empty_round": not bool(cl.size),
        "aggregation_norm_mean": 0.0,
        "aggregation_norm_var": 0.0,
        "aggregation_norm_max": 0.0,
        "aggregation_norm_median": 0.0,
        "aggregation_norm_ratio": 0.0,
        "aggregation_sources": [],
        "using_robust": bool(use_robust),
        "decision_interval": 1,
        "client_change_monitor_ready": False,
        "client_model_score_drop_p50": 0.0,
        "client_model_score_drop_p90": 0.0,
        "client_update_direction_dispersion": 0.0,
        "online_roster_overlap": 0.0,
        "client_score_exceedance_fraction": 0.0,
        "client_score_exceedance_overlap": 0.0,
        "client_update_group_separation": 0.0,
        "client_loss_mean_delta": 0.0,
        "client_utilities": [],
        "select_k": 1,
        "select_enabled": False,
        "select_seed": 0,
        "oort_exploration": 0.0,
        "oort_round_threshold": 0.0,
        "oort_prefer_duration": 0.0,
        "oort_sample_window": 1,
    }


def render_text(obs):
    participation = (
        f"Planned participation {obs['planned_participation_rate']:.2f}, "
        f"availability {obs['availability_rate']:.2f}, "
        f"participation gap {obs['participation_gap']:.2f}. "
        if "planned_participation_rate" in obs else
        f"Participation rate {obs['participation_rate']:.2f}. "
    )
    return (
        f"Round {obs['round']}/{obs['total_rounds']}. "
        f"Global accuracy {obs['global_acc']:.3f} "
        f"(3-round change {obs['acc_delta_3']:+.3f}). "
        f"Client loss mean {obs['client_loss_mean']:.3f}, variance {obs['client_loss_var']:.3f}. "
        f"Client update-norm mean {obs['update_norm_mean']:.3f}, variance {obs['update_norm_var']:.4f}. "
        f"Update-norm max/median {obs['update_norm_ratio']:.1f} "
        f"(~1.5+ = a few clients' updates far above the rest; ~1.1 = clients move together). "
        f"Input shift {obs['input_shift']:.2f}, label-dist shift {obs['label_shift']:.2f} "
        f"(input shift high = inputs P(x) moved; label shift high = roster-normalized class mix P(y) changed). "
        f"Current learning rate {obs['current_lr']:.4f}. "
        f"{participation}"
        f"Robust aggregation: {'on' if obs['using_robust'] else 'off'}."
    )
