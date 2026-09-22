import sys
import json
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_decision_audit import (
    audit_episode, agent_config_id, summarize_agents,
)
from llm_backend import load_model_specs


def _manifest():
    actions = {
        "no_op": {"family": "control"},
        "drift_adapt": {"family": "data_adaptation"},
        "dropout_handle": {"family": "missing_client_handling"},
        "friend_substitute": {"family": "missing_client_handling"},
        "robust": {"family": "aggregation_defense"},
        "spawn_concept_auto": {"family": "concept_routing"},
    }
    return {
        "actions": actions,
        "causes": {
            "real_drift": {
                "feasible_actions": ["drift_adapt"],
                "canonical_action": "drift_adapt",
            },
            "dropout": {
                "feasible_actions": ["dropout_handle", "friend_substitute"],
                "canonical_action": "dropout_handle",
            },
            "fault": {
                "feasible_actions": ["robust"],
                "canonical_action": "robust",
            },
            "staggered_concepts": {
                "feasible_actions": ["spawn_concept_auto"],
                "canonical_action": "spawn_concept_auto",
            },
        },
        "protocol": {"response_window_length": 5},
    }


def _telemetry(total_rounds, bundles, invalid=None, interface_error=None):
    invalid = invalid or {}
    interface_error = interface_error or {}
    rows = []
    for round_idx in range(total_rounds):
        bundle = bundles.get(round_idx, ("no_op",))
        status = invalid.get(round_idx, "VALID")
        rows.append({
            "round": round_idx,
            "decision_due": True,
            "decision_bundle_status": status,
            "declared_bundle": "|".join(bundle) if status == "VALID" else "",
            "raw_output": json.dumps({"actions": list(bundle)}),
            "public_observation_json": json.dumps({"round": round_idx}),
            "agent_interface_error": interface_error.get(round_idx, ""),
            "llm_api_error": interface_error.get(round_idx, ""),
            "llm_budget_error": "",
        })
    return rows


def _complex_plan():
    return {
        "episode_id": "ep_demo",
        "case_id": "complex",
        "training_seed": 0,
        "partition_mode": "noniid",
        "total_rounds": 80,
        "hidden_event": {
            "cause_intervals": {
                "real_drift": {"0": [[40, 70]]},
                "dropout": {"1": [[40, 50]]},
                "fault": {"2": [[50, 60]]},
            },
            "affected_clients": {
                "real_drift": {"clients": [0]},
                "dropout": {"clients": [1]},
                "fault": {"clients": [2]},
            },
            "event_schedule": [
                {"round": 40, "activate": ["real_drift", "dropout"],
                 "severity": {}, "emit_deadline": 44},
                {"round": 50, "activate": ["fault"],
                 "severity": {}, "emit_deadline": 52},
                {"round": 70, "activate": [],
                 "severity": {}, "emit_deadline": 71},
            ],
        },
    }


def _spec(name="demo"):
    return {"name": name, "model": "fixture", "decision_cadence": 1}


def test_complex_audit_matches_planned_example():
    bundles = {round_idx: ("no_op",) for round_idx in range(80)}
    bundles.update({
        40: ("no_op",), 41: ("robust",), 42: ("no_op",), 43: ("no_op",),
        44: ("drift_adapt",),
        45: ("drift_adapt", "dropout_handle", "robust"),
        46: ("drift_adapt", "friend_substitute"), 47: ("no_op",),
        48: ("drift_adapt", "dropout_handle"),
        49: ("drift_adapt", "dropout_handle"),
        50: ("drift_adapt", "dropout_handle"),
    })
    for round_idx in range(51, 60):
        bundles[round_idx] = ("drift_adapt", "robust")
    for round_idx in range(60, 62):
        bundles[round_idx] = ("drift_adapt", "robust")
    for round_idx in range(62, 70):
        bundles[round_idx] = ("drift_adapt",)
    for round_idx in range(70, 80):
        bundles[round_idx] = ("drift_adapt",)
    invalid = {42: "CONFLICT", 43: "UNKNOWN_ACTION", 47: "INVALID_SCHEMA"}
    result = audit_episode(
        "run_demo", "demo", _spec(), _complex_plan(),
        {"telemetry": _telemetry(80, bundles, invalid)}, _manifest(),
    )
    assert result["integrity"]["run_status"] == "VALID"
    assert [row["transition_type"] for row in result["events"]] == [
        "INIT", "ACTIVATE", "CHANGE", "CHANGE", "STABLE",
    ]
    e01, e02, e03, e04 = result["events"][1:]
    assert (e01["correct_decision_round"], e01["timing_status"],
            e01["invalid_bundle_count"], e01["post_correct_regression_count"]) == (
                46, "LATE", 3, 1,
            )
    assert e02["removed_causes"] == "dropout" and e02["timing_status"] == "ON_TIME"
    assert e03["removed_causes"] == "fault" and e03["decision_delay"] == 2
    assert e04["event_outcome_bucket"] == "NONE_BEST"
    config_id = agent_config_id(_spec())
    summary = summarize_agents(result["events"], result["decisions"], {
        config_id: _spec(),
    })[0]
    assert summary["evaluated_events"] == 4
    assert summary["ever_correct_event_rate"] == 0.75
    assert summary["correct_decision_rate"] == 0.5
    assert summary["invalid_bundle_rate"] == 0.075
    assert summary["over_intervention_decision_rate"] == 0.375
    assert summary["cause_removal_event_count"] == 3


def test_full_coverage_with_extra_has_own_event_bucket():
    plan = _complex_plan()
    bundles = {round_idx: ("no_op",) for round_idx in range(80)}
    for round_idx in range(40, 50):
        bundles[round_idx] = ("drift_adapt", "dropout_handle", "robust")
    result = audit_episode(
        "run_demo", "extra", _spec("extra"), plan,
        {"telemetry": _telemetry(80, bundles)}, _manifest(),
    )
    event = result["events"][1]
    assert event["best_coverage"] == "FULL"
    assert event["never_correct"] is True
    assert event["event_outcome_bucket"] == "FULL_WITH_EXTRA_BEST"


def test_one_shot_completion_persists_across_reactivation():
    manifest = _manifest()
    manifest["protocol"] = {}
    plan = {
        "episode_id": "ep_one_shot", "case_id": "one_shot", "training_seed": 0,
        "partition_mode": "noniid", "total_rounds": 9,
        "hidden_event": {
            "cause_intervals": {"staggered_concepts": {"0": [[2, 5], [6, 8]]}},
            "affected_clients": {"staggered_concepts": {"clients": [0]}},
            "event_schedule": [
                {"round": 2, "activate": ["staggered_concepts"], "severity": {}},
                {"round": 6, "activate": ["staggered_concepts"], "severity": {}},
            ],
        },
    }
    bundles = {round_idx: ("no_op",) for round_idx in range(9)}
    bundles[2] = ("spawn_concept_auto",)
    bundles[7] = ("spawn_concept_auto",)
    result = audit_episode(
        "run_demo", "one_shot", _spec("one_shot"), plan,
        {"telemetry": _telemetry(9, bundles)}, manifest,
    )
    reactivated = result["events"][3]
    assert reactivated["canonical_bundle"] == "no_op"
    assert reactivated["correct_decision_round"] == 6
    repeated = next(row for row in result["decisions"] if row["decision_round"] == 7)
    assert repeated["unnecessary_actions"] == "spawn_concept_auto"
    assert repeated["decision_label"] == "OVER_INTERVENTION"


def test_interface_failure_invalidates_whole_agent_episode():
    plan = _complex_plan()
    result = audit_episode(
        "run_demo", "broken", _spec("broken"), plan,
        {"telemetry": _telemetry(80, {}, interface_error={41: "TimeoutError"})},
        _manifest(),
    )
    assert result["integrity"]["run_status"] == "INVALID"
    assert result["integrity"]["failure_type"] == "AGENT_INTERFACE_FAILED"
    assert result["events"] == [] and result["decisions"] == []


def test_scheduled_startup_cause_gets_a_response_deadline():
    plan = {
        "episode_id": "ep_startup", "case_id": "startup", "training_seed": 0,
        "partition_mode": "noniid", "total_rounds": 12,
        "hidden_event": {
            "cause_intervals": {"real_drift": {"0": [[0, None]]}},
            "affected_clients": {"real_drift": {"clients": [0]}},
            "event_schedule": [
                {"round": 0, "activate": ["real_drift"], "severity": {}},
            ],
        },
    }
    bundles = {round_idx: ("no_op",) for round_idx in range(12)}
    bundles[3] = ("drift_adapt",)
    result = audit_episode(
        "run_demo", "startup", _spec("startup"), plan,
        {"telemetry": _telemetry(12, bundles)}, _manifest(),
    )
    event = result["events"][0]
    assert event["transition_type"] == "INIT"
    assert event["emit_deadline"] == 4
    assert event["timing_status"] == "ON_TIME"


def test_deadline_must_precede_transition_and_have_a_call_opportunity():
    crossing = _complex_plan()
    crossing["hidden_event"]["event_schedule"][0]["emit_deadline"] = 50
    telemetry = _telemetry(80, {})
    try:
        audit_episode(
            "run_demo", "crossing", _spec("crossing"), crossing,
            {"telemetry": telemetry}, _manifest(),
        )
    except ValueError as exc:
        assert "crosses the next transition" in str(exc)
    else:
        raise AssertionError("deadline crossing the next transition was accepted")

    no_opportunity = deepcopy(_complex_plan())
    for row in telemetry:
        if 40 <= row["round"] <= 44:
            row["decision_due"] = False
    try:
        audit_episode(
            "run_demo", "no_opportunity", _spec("no_opportunity"), no_opportunity,
            {"telemetry": telemetry}, _manifest(),
        )
    except ValueError as exc:
        assert "no call before deadline" in str(exc)
    else:
        raise AssertionError("deadline without an Agent call opportunity was accepted")


def test_formal_model_roster_contains_four_configured_models():
    models = load_model_specs(Path(__file__).parents[1] / "src" / "benchmark_models.json", require_keys=False)
    assert [model["model"] for model in models] == [
        "qwen3.8-max", "qwen3.8-flash", "kimi-k3", "deepseek-v4-pro-0813",
    ]
    assert len({model["name"] for model in models}) == 4


if __name__ == "__main__":
    test_complex_audit_matches_planned_example()
    test_full_coverage_with_extra_has_own_event_bucket()
    test_one_shot_completion_persists_across_reactivation()
    test_interface_failure_invalidates_whole_agent_episode()
    test_scheduled_startup_cause_gets_a_response_deadline()
    test_deadline_must_precede_transition_and_have_a_call_opportunity()
    test_formal_model_roster_contains_four_configured_models()
    print("AGENT_DECISION_AUDIT_OK")
