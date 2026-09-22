"""Agent-visible observation contract; stdlib-only for audit imports."""

import json
import math


PUBLIC_OBSERVATION_SCHEMA = {
    "round": "integer",
    "total_rounds": "integer",
    "global_acc": "number",
    "acc_delta_3": "number",
    "client_loss_mean": "number",
    "client_loss_var": "number",
    "update_norm_mean": "number",
    "update_norm_var": "number",
    "update_norm_max": "number",
    "update_norm_median": "number",
    "update_norm_ratio": "number",
    "input_shift": "number",
    "label_shift": "number",
    "current_lr": "number",
    "participation_rate": "number",
    "planned_participation_rate": "number",
    "availability_rate": "number",
    "participation_gap": "number",
    "active_actions": "array",
    "empty_round": "boolean",
    "aggregation_norm_mean": "number",
    "aggregation_norm_var": "number",
    "aggregation_norm_max": "number",
    "aggregation_norm_median": "number",
    "aggregation_norm_ratio": "number",
    "aggregation_sources": "array",
    "using_robust": "boolean",
    "decision_interval": "integer",
    "client_change_monitor_ready": "boolean",
    "client_model_score_drop_p50": "number",
    "client_model_score_drop_p90": "number",
    "client_update_direction_dispersion": "number",
    "online_roster_overlap": "number",
    "client_score_exceedance_fraction": "number",
    "client_score_exceedance_overlap": "number",
    "client_update_group_separation": "number",
    "client_loss_mean_delta": "number",
    "client_utilities": "array",
    "select_k": "integer",
    "select_enabled": "boolean",
    "select_seed": "integer",
    "oort_exploration": "number",
    "oort_round_threshold": "number",
    "oort_prefer_duration": "number",
    "oort_sample_window": "integer",
}

PRIVATE_SCORER_SCHEMA = {
    "episode_id": "string",
    "planned_clients_count": "integer",
    "available_clients_count": "integer",
    "participating_clients_count": "integer",
    "dropout_removed_clients": "integer",
    "data_shift_injected_clients": "integer",
    "fault_injected_clients": "integer",
    "robust_filtered_clients": "integer",
    "drifted_client_count": "integer",
    "cluster_split_score": "number",
    "feddrift_trigger": "boolean",
    "feddrift_loss_delta": "number",
    "trusted_cache_count": "integer",
    "cache_rejection_count": "integer",
    "max_cache_age": "integer",
}


def public_observation(values: dict) -> dict:
    missing = set(PUBLIC_OBSERVATION_SCHEMA) - set(values)
    extra = set(values) - set(PUBLIC_OBSERVATION_SCHEMA)
    if missing or extra:
        raise ValueError(f"public observation schema mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for key, kind in PUBLIC_OBSERVATION_SCHEMA.items():
        value = values[key]
        valid = {
            "integer": type(value) is int,
            "number": type(value) in {int, float} and math.isfinite(float(value)),
            "boolean": type(value) is bool,
            "array": isinstance(value, (list, tuple)),
        }[kind]
        if not valid:
            raise ValueError(f"public observation field {key!r} is not {kind}")
    return {key: values[key] for key in PUBLIC_OBSERVATION_SCHEMA}


def serialize_public_observation(obs: dict) -> str:
    return json.dumps(public_observation(obs), sort_keys=True, separators=(",", ":"))


def llm_observation(obs: dict, decision_memory=None) -> dict:
    """Compact controller view; full public telemetry remains available to tools/audit."""
    view = public_observation(obs)
    utilities = view.pop("client_utilities")
    for key in (
        "select_k", "select_enabled", "select_seed",
        "oort_exploration", "oort_round_threshold", "oort_prefer_duration",
        "oort_sample_window",
    ):
        view.pop(key)
    scores = [float(row["utility"]) for row in utilities
              if isinstance(row, dict) and type(row.get("utility")) in {int, float}
              and math.isfinite(float(row["utility"]))]
    mean = math.fsum(scores) / len(scores) if scores else 0.0
    variance = (math.fsum((score - mean) ** 2 for score in scores) / len(scores)
                if scores else 0.0)
    view.update({
        "input_moment_delta": view.pop("input_shift"),
        "aggregate_label_hist_tv": view.pop("label_shift"),
        "utility_client_count": len(utilities),
        "utility_mean": round(mean, 6),
        "utility_std": round(math.sqrt(variance), 6),
        "utility_max": round(max(scores), 6) if scores else 0.0,
    })
    view.update(decision_memory or {
        "last_decision": None,
        "active_action_ages": {},
        "telemetry_delta_since_last_decision": {},
    })
    return view


ABLATION_FIELDS = {
    "G1": {"global_acc", "acc_delta_3", "client_loss_mean", "client_loss_var", "client_loss_mean_delta"},
    "G2": {"update_norm_mean", "update_norm_var", "update_norm_max", "update_norm_median", "update_norm_ratio", "aggregation_norm_mean", "aggregation_norm_var", "aggregation_norm_max", "aggregation_norm_median", "aggregation_norm_ratio", "client_change_monitor_ready", "client_model_score_drop_p50", "client_model_score_drop_p90", "client_update_direction_dispersion", "client_score_exceedance_fraction", "client_score_exceedance_overlap", "client_update_group_separation"},
    "G3": {"input_shift", "label_shift", "input_moment_delta", "aggregate_label_hist_tv"},
    "G4": {"participation_rate", "planned_participation_rate", "availability_rate", "participation_gap", "online_roster_overlap", "empty_round", "aggregation_sources", "client_utilities", "select_k", "select_enabled", "select_seed", "oort_exploration", "oort_round_threshold", "oort_prefer_duration", "oort_sample_window", "utility_client_count", "utility_mean", "utility_std", "utility_max"},
    "G5": {"active_actions", "using_robust", "last_decision", "active_action_ages", "telemetry_delta_since_last_decision"},
}


def ablate_observation(view: dict, ablation_group=None) -> dict:
    if not ablation_group:
        return view
    if ablation_group not in ABLATION_FIELDS:
        raise ValueError(f"unknown ablation group: {ablation_group}")
    masked = dict(view)
    fields = ABLATION_FIELDS[ablation_group]
    for key in fields:
        masked.pop(key, None)
    deltas = masked.get("telemetry_delta_since_last_decision")
    if isinstance(deltas, dict):
        masked["telemetry_delta_since_last_decision"] = {
            key: value for key, value in deltas.items() if key not in fields
        }
    return masked


def serialize_llm_observation(obs: dict, decision_memory=None, ablation_group=None) -> str:
    view = ablate_observation(llm_observation(obs, decision_memory), ablation_group)
    return json.dumps(
        view, sort_keys=True, separators=(",", ":")
    )


def private_scorer_record(values: dict) -> dict:
    missing = set(PRIVATE_SCORER_SCHEMA) - set(values)
    extra = set(values) - set(PRIVATE_SCORER_SCHEMA)
    if missing or extra:
        raise ValueError(f"private scorer schema mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for key, kind in PRIVATE_SCORER_SCHEMA.items():
        value = values[key]
        valid = {
            "string": type(value) is str,
            "integer": type(value) is int,
            "number": type(value) in {int, float} and math.isfinite(float(value)),
            "boolean": type(value) is bool,
        }[kind]
        if not valid:
            raise ValueError(f"private scorer field {key!r} is not {kind}")
    return {key: values[key] for key in PRIVATE_SCORER_SCHEMA}
