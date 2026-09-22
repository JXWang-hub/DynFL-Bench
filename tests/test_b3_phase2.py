"""Small runnable check for the Phase 2 ActionBundle contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import tempfile
import torch
import b3_composite as composite_module
import engine as engine_module
from types import SimpleNamespace

from agents import (
    ACTION_FAMILY,
    B3_VISIBLE_ACTIONS,
    Action,
    ActionBundle,
    Agent,
    CompositeDiagnoseAgent,
    CompositeOracleAgent,
    LLMAgent,
    NoOpAgent,
    RandomBundleV2Agent,
    SelectClientsAgent,
    as_action_bundle,
)
from action_contract import ActionState, reduce_action_state
from runtime_contract import (ClientRuntimeState, FedDAAReport, FedDriftReport,
                              LabelHistogramReport, LabelShiftReport,
                              UpdateEnvelope)
from observation_contract import (
    PUBLIC_OBSERVATION_SCHEMA,
    ablate_observation,
    serialize_llm_observation,
)
from data import dirichlet_partition, iid_partition, partition_sha256
from episode_timeline import EpisodeTimeline, compile_episode_timeline
from engine import Engine
from b3_composite import _checkpoint_history, prepare_episode
from b3_episode_plan import build_episode_plan
from llm_backend import B3_SYSTEM_PROMPT
from multi_seed import make_cfg
from run import build_data, set_seed
from run_rule_telemetry_pilot import extra_action_rounds
from scoring import summarize
from telemetry import collect, norm_stats, render_text


def expect_invalid(*actions):
    try:
        ActionBundle(actions)
    except ValueError:
        return
    raise AssertionError(f"expected invalid bundle: {actions}")


class BundleAgent(Agent):
    name = "bundle_check"

    def __init__(self):
        self.fired = False

    def decide(self, obs):
        if not self.fired:
            self.fired = True
            return ActionBundle(("robust", "dropout_handle"))
        return Action("no_op")


class StaleAgent(Agent):
    name = "stale_check"
    always_decide = True

    def decide(self, obs):
        return Action("dropout_handle")


class PersistentBundleAgent(Agent):
    always_decide = True

    def __init__(self, *actions):
        self.actions = actions

    def decide(self, obs):
        return ActionBundle(self.actions)


class SelectOnceAgent(Agent):
    def decide(self, obs):
        if obs["round"] == 0:
            return Action("select_clients", (2, 3))
        return Action("no_op")


def main():
    sample = {
        "global_acc": 0.8,
        "update_norm_ratio": 2.0,
        "client_change_monitor_ready": True,
        "input_shift": 2.0,
        "label_shift": 0.3,
        "participation_gap": 0.5,
        "empty_round": True,
        "aggregation_sources": ["online"],
        "client_utilities": [{"client": 0, "utility": 1.0}],
        "select_enabled": True,
        "telemetry_delta_since_last_decision": {
            "global_acc": -0.1,
            "update_norm_ratio": 0.4,
            "participation_gap": 0.5,
        },
    }
    g1 = ablate_observation(sample, "G1")
    assert "global_acc" not in g1
    assert "global_acc" not in g1["telemetry_delta_since_last_decision"]
    g2 = ablate_observation(sample, "G2")
    assert "update_norm_ratio" not in g2
    assert "update_norm_ratio" not in g2["telemetry_delta_since_last_decision"]
    assert "client_change_monitor_ready" not in g2
    g3 = ablate_observation(sample, "G3")
    assert "input_shift" not in g3 and "label_shift" not in g3
    g4 = ablate_observation(sample, "G4")
    assert not ({"participation_gap", "empty_round", "aggregation_sources",
                 "client_utilities", "select_enabled"} & set(g4))
    assert "participation_gap" not in g4["telemetry_delta_since_last_decision"]
    assert "telemetry_delta_since_last_decision" not in ablate_observation(sample, "G5")
    try:
        ablate_observation(sample, "G0")
        raise AssertionError("unknown ablation group accepted")
    except ValueError:
        pass
    fixture_model = {
        "name": "fixture", "kind": "fixture", "model": "fixture-model",
        "temperature": 0.0, "base_url": "https://invalid.example",
        "api_key_env": "FIXTURE_KEY", "input_cost_per_million": None,
        "output_cost_per_million": None, "reasoning_effort": "low",
    }
    fixture_policy = {
        "max_calls": 80, "decision_cadence": 1, "timeout_seconds": 1.0,
        "retry_delays_seconds": [],
    }
    normal_agent, normal_spec = composite_module._configured_llm_agent(
        fixture_model, fixture_policy, "fixture-sha", query_fn=lambda _: "{}",
    )
    assert "ablation_group" not in normal_spec
    assert not hasattr(normal_agent, "ablation_group")
    ablated_agent, ablated_spec = composite_module._configured_llm_agent(
        fixture_model, fixture_policy, "fixture-sha", query_fn=lambda _: "{}",
        ablation_group="G1",
    )
    assert ablated_spec["ablation_group"] == "G1"
    assert ablated_agent.ablation_group == "G1"
    dropout_obs = {
        "round": 40, "participation_gap": 0.5, "availability_rate": 0.5,
        "participation_rate": 0.5, "planned_participation_rate": 1.0,
        "select_enabled": False,
    }
    assert CompositeDiagnoseAgent().decide(dropout_obs).type == "dropout_handle"
    assert CompositeDiagnoseAgent(ablation_group="G4").decide(dropout_obs).type == "no_op"

    assert extra_action_rounds(
        {40: {"moment_align_adapt"},
         42: {"moment_align_adapt", "spawn_concept_auto"}},
        {"moment_align_adapt", "robust"},
    ) == [42]
    labels = torch.arange(100) % 10
    iid_a = iid_partition(labels, 5, 10, 7)
    iid_b = iid_partition(labels, 5, 10, 7)
    assert partition_sha256(iid_a) == partition_sha256(iid_b)
    assert partition_sha256(iid_a, "iid") != partition_sha256(iid_a, "noniid")
    assert len({int(index) for client in iid_a for index in client}) == 50

    episode_id = "ep_" + "8" * 32
    noniid_plan = build_episode_plan("real_dropout", 0, episode_id=episode_id)
    iid_plan = build_episode_plan(
        "real_dropout", 0, episode_id=episode_id, partition_mode="iid"
    )
    assert iid_plan["hidden_event"] == noniid_plan["hidden_event"]
    assert iid_plan["partition_mode"] == "iid"
    hetero_iid = build_episode_plan(
        "hetero_abrupt_dropout", 0, partition_mode="iid"
    )
    assert hetero_iid["workpoint_role"] == "iid_system_control"
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = Path(tmp) / "paired.json"
        prepare_episode(
            "real_dropout", 0, True, True, plan_path=plan_path,
            partition_mode="noniid",
        )
        try:
            prepare_episode(
                "real_dropout", 0, True, True, plan_path=plan_path, resume=True,
                partition_mode="iid",
            )
        except ValueError as exc:
            assert "version/fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("cross-partition private-plan resume did not fail closed")

    manifest = json.loads((Path(__file__).parents[1] / "src" / "b3_suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["response_window_length"] == 10
    assert tuple(manifest["actions"]) == B3_VISIBLE_ACTIONS
    assert all(ACTION_FAMILY[name] == row["family"] for name, row in manifest["actions"].items())

    bundle = ActionBundle(("robust", "dropout_handle", "dropout_handle"))
    assert bundle.action_types == ("robust", "dropout_handle")
    assert tuple(action.type for action in bundle.ordered_actions) == (
        "dropout_handle", "robust"
    )
    assert as_action_bundle(Action("drift_adapt")).type == "drift_adapt"
    expect_invalid("no_op", "robust")
    expect_invalid("drift_adapt", "moment_align_adapt")
    expect_invalid("drift_adapt", "robust", "dropout_handle", "select_clients")

    random_v2 = RandomBundleV2Agent(p_act=1.0, seed=7, max_actions=3)
    sampled_sizes = set()
    for _ in range(100):
        sampled = as_action_bundle(random_v2.decide({}))
        sampled_sizes.add(len(sampled.actions))
        families = [ACTION_FAMILY[name] for name in sampled.action_types]
        assert len(families) == len(set(families))
    assert sampled_sizes == {1, 2, 3}
    assert RandomBundleV2Agent(p_act=0.0).decide({}).type == "no_op"
    phase5_names = {
        agent.name for agent in composite_module._phase5_roster(
            0, 40, ("drift_adapt", "robust"),
        )
    }
    assert "random_bundle" not in phase5_names
    assert "random_bundle_v2" not in phase5_names

    state = ActionState()
    first = reduce_action_state(state, ActionBundle(("robust",)))
    assert first.active_before == () and first.active_after == ("robust",)
    switched = reduce_action_state(first.state, ActionBundle(("moment_align_adapt",)))
    assert switched.active_after == ("moment_align_adapt",)
    assert switched.started == ("moment_align_adapt",) and switched.stopped == ("robust",)
    closed = reduce_action_state(switched.state, ActionBundle(("no_op",)))
    assert closed.active_after == () and closed.stopped == ("moment_align_adapt",)
    state = ActionState()
    for index in range(100):
        transition = reduce_action_state(state, ActionBundle(("drift_adapt",)))
        assert transition.active_after == ("drift_adapt",)
        assert transition.started == (("drift_adapt",) if index == 0 else ())
        state = transition.state

    runtime = ClientRuntimeState()
    clean = UpdateEnvelope(1, 3, "online_real", 1, "unverified", {"w": 1}, 8)
    assert runtime.usable_cache(3, 2) is None
    runtime.accept(clean)
    assert runtime.usable_cache(5, 2).source_round == 3
    assert runtime.usable_cache(6, 2) is None and runtime.rejection_reason == "cache_expired"
    assert runtime.trusted_cache is None
    runtime.accept(clean)
    runtime.reject("robust_rejected_or_unverifiable")
    assert runtime.trusted_cache is None and runtime.usable_cache(4, 2) is None
    runtime.observe_availability(False, 4, selected=True)
    runtime.observe_availability(False, 5, selected=True)
    runtime.observe_availability(False, 6, selected=True)
    assert runtime.unavailable_streak == 3 and runtime.cooldown_until == 9
    runtime.observe_availability(True, 7, selected=True)
    assert runtime.return_state == "returned" and runtime.returned_round == 7
    assert runtime.cooldown_until == 7 and runtime.selection_attempts == 4
    report = FedDriftReport(1, 7, 8, ((2, 3, 0.75), (4, 1, 0.5)))
    assert report.scores() == {2: 0.75, 4: 0.5}
    try:
        FedDriftReport(1, 7, 8, ((2, 3, float("nan")),))
        raise AssertionError("non-finite FedDrift score accepted")
    except ValueError:
        pass
    label_report = LabelHistogramReport(1, 7, (2, 3, 3))
    shift_report = LabelShiftReport(1, 7, (3, 3, 2), (2, 3, 3))
    feddaa_report = FedDAAReport(
        1, 7, 8, 2, (0.8, 0.2, 0.1, 0.9), (3, 5), ((0, 0.4),)
    )
    assert sum(label_report.class_counts) == 8
    assert sum(shift_report.baseline_counts) == sum(shift_report.current_counts) == 8
    assert feddaa_report.losses() == {0: 0.4} and feddaa_report.communication_bytes == 104
    try:
        FedDAAReport(1, 7, 8, 2, (0.8, 0.2, 0.1, float("nan")),
                     (3, 5), ((0, 0.4),))
        raise AssertionError("non-finite FedDAA prototype accepted")
    except ValueError:
        pass

    legacy_cfg = SimpleNamespace(
        scenario="drift", drift_type="real", drift_round=4,
        drift_mode="recurrent", recur_gap=3, dropout_recover_round=0,
    )
    legacy_time = compile_episode_timeline(
        legacy_cfg, {"drift_schedule": {0: 4, 1: 5}}
    )
    assert legacy_time.event_round() == 4
    assert legacy_time.active("real_drift", 4, 0)
    assert not legacy_time.active("real_drift", 4, 1)
    assert not legacy_time.active("real_drift", 7, 0)
    private_time = compile_episode_timeline(legacy_cfg, {
        "drift_schedule": {0: 4, 1: 4},
        "b3_episode_plan": {"hidden_event": {
            "active_causes": ["real_drift"],
            "affected_clients": {"real_drift": {"clients": [0, 1]}},
            "event_schedule": [{"round": 4, "activate": ["real_drift"],
                                "severity": {"real_drift": {}}}],
        }},
    })
    assert private_time.event_round() == 4
    assert private_time.active("real_drift", 4, 0)

    intervals = EpisodeTimeline(("dropout",), 40, {"dropout": {
        0: ((40, 50),), 1: ((45, None),), 2: ((10, 20), (55, None)),
    }})
    assert intervals.active("dropout", 49, 0)
    assert not intervals.active("dropout", 50, 0)
    assert intervals.active("dropout", 45, 1)
    assert intervals.active("dropout", 15, 2) and intervals.active("dropout", 55, 2)

    labels = torch.tensor(list(range(10)) * 10)
    parts = dirichlet_partition(labels, 4, 0.3, 20, 7)
    flattened = [int(index) for part in parts for index in part]
    assert len(flattened) == 80 and len(set(flattened)) == 80
    repeat = dirichlet_partition(labels, 4, 0.3, 20, 7)
    assert partition_sha256(parts) == partition_sha256(repeat)

    shared_hashes = []
    for scenario in ("drift", "fault", "dropout"):
        shared_cfg = make_cfg(scenario, 11, synthetic=True, smoke=True)
        shared_cfg.num_clients = 4
        shared_cfg.samples_per_client = 20
        shared_cfg.test_size = 50
        shared_hashes.append(build_data(shared_cfg)["partition_sha256"])
    assert len(set(shared_hashes)) == 1

    public_histories = []
    for scenario in ("drift", "fault", "dropout"):
        shared_cfg = make_cfg(scenario, 12, synthetic=True, smoke=True)
        shared_cfg.num_clients = 4
        shared_cfg.samples_per_client = 20
        shared_cfg.test_size = 50
        shared_cfg.rounds = 2
        shared_cfg.drift_round = 3
        set_seed(shared_cfg.seed)
        rows = Engine(shared_cfg, build_data(shared_cfg)).run(NoOpAgent(), log=False)["telemetry"]
        public_histories.append([
            {key: row[key] for key in PUBLIC_OBSERVATION_SCHEMA} for row in rows
        ])
    assert public_histories[0] == public_histories[1] == public_histories[2]
    serialized_g4 = json.loads(serialize_llm_observation(
        public_histories[0][0],
        {"telemetry_delta_since_last_decision": {"participation_gap": 0.5}},
        "G4",
    ))
    assert not ({"participation_gap", "empty_round", "aggregation_sources"} &
                set(serialized_g4))
    assert serialized_g4["telemetry_delta_since_last_decision"] == {}

    universal_cfg = make_cfg("dropout", 14, synthetic=True, smoke=True)
    universal_cfg.num_clients = 4
    universal_cfg.samples_per_client = 20
    universal_cfg.test_size = 50
    universal_cfg.rounds = 2
    universal_cfg.drift_round = 3
    feddrift_calls = []
    original_update_reports = engine_module._feddrift_update_reports
    def recording_update_reports(reports, *args):
        feddrift_calls.append(tuple(report.client_id for report in reports))
        return original_update_reports(reports, *args)
    engine_module._feddrift_update_reports = recording_update_reports
    try:
        set_seed(universal_cfg.seed)
        Engine(universal_cfg, build_data(universal_cfg)).run(NoOpAgent(), log=False)
    finally:
        engine_module._feddrift_update_reports = original_update_reports
    assert feddrift_calls == [(0, 1, 2, 3), (0, 1, 2, 3)]

    ordered_cfg = make_cfg("dropout", 13, synthetic=True, smoke=True)
    ordered_cfg.num_clients = 4
    ordered_cfg.samples_per_client = 20
    ordered_cfg.test_size = 50
    ordered_cfg.rounds = 2
    ordered_data = build_data(ordered_cfg)
    ordered_data["b3_episode_plan"] = {
        "episode_id": "ep_" + "1" * 32,
        "plan_sha256": "fixture",
        "partition_sha256": ordered_data["partition_sha256"],
        "hidden_event": {
            "active_causes": ["fault", "dropout"],
            "affected_clients": {
                "fault": {"clients": [0, 1, 2]},
                "dropout": {"clients": [0, 1, 2]},
            },
            "event_schedule": [
                {"round": 0, "activate": ["fault", "dropout"], "severity": {}},
            ],
            "cause_intervals": {
                "fault": {"0": [[0, 1]], "1": [[1, None]], "2": [[1, 2]]},
                "dropout": {"0": [[1, None]], "1": [[0, 1]], "2": [[1, 2]]},
            },
        },
    }
    set_seed(ordered_cfg.seed)
    ordered_rows = Engine(ordered_cfg, ordered_data).run(NoOpAgent(), log=False)["telemetry"]
    assert ordered_rows[0]["fault_injected_clients"] == 1
    assert ordered_rows[0]["available_clients_count"] == 3
    assert ordered_rows[1]["fault_injected_clients"] == 1
    assert ordered_rows[1]["available_clients_count"] == 2

    selected_cfg = make_cfg("dropout", 15, synthetic=True, smoke=True)
    selected_cfg.num_clients = 4
    selected_cfg.samples_per_client = 20
    selected_cfg.test_size = 50
    selected_cfg.rounds = 2
    selected_cfg.participation = 0.5
    selected_data = build_data(selected_cfg)
    selected_data["b3_episode_plan"] = {
        "episode_id": "ep_" + "2" * 32,
        "plan_sha256": "fixture",
        "partition_sha256": selected_data["partition_sha256"],
        "hidden_event": {
            "active_causes": ["dropout"],
            "affected_clients": {"dropout": {"clients": [0, 1]}},
            "event_schedule": [
                {"round": 1, "activate": ["dropout"], "severity": {}},
            ],
            "cause_intervals": {
                "dropout": {"0": [[1, None]], "1": [[1, None]]},
            },
        },
    }
    set_seed(selected_cfg.seed)
    selected_rows = Engine(selected_cfg, selected_data).run(
        SelectOnceAgent(), log=False
    )["telemetry"]
    assert selected_rows[0]["availability_rate"] == 1.0
    assert selected_rows[1]["planned_participation_rate"] == 0.5
    assert selected_rows[1]["availability_rate"] == 0.5
    assert selected_rows[1]["participation_gap"] == 0.0
    ordered_agent = CompositeDiagnoseAgent()
    first_diagnosis = ordered_agent.decide({
        **ordered_rows[0], "update_norm_ratio": 3.0,
    })
    assert "robust" not in first_diagnosis.action_types
    diagnosed = ordered_agent.decide({
        **ordered_rows[1], "update_norm_ratio": 3.0,
    })
    assert {"robust", "dropout_handle"} <= set(diagnosed.action_types), (
        diagnosed.action_types,
    )

    obs = collect(
        40, 0.5, [0.6, 0.55, 0.5, 0.5], [1.0, 1.2], [1.0, 1.1],
        0.01, 0.5, False, 80,
    )
    obs.update({
        "planned_participation_rate": 1.0,
        "availability_rate": 0.5,
        "participation_gap": 0.5,
        "client_loss_mean_delta": 0.2,
        "label_shift": 0.10,
        "client_score_exceedance_fraction": 0.70,
        "select_enabled": False,
    })
    real_rule = CompositeDiagnoseAgent()
    assert real_rule.decide(obs).action_types == ("dropout_handle",)
    rule = real_rule.decide({**obs, "round": 41})
    assert rule.action_types == ("drift_adapt", "dropout_handle")
    masked = {
        **obs, "acc_delta_3": -0.5, "label_shift": 0.3,
        "client_score_exceedance_fraction": 0.0,
    }
    masked_rule = CompositeDiagnoseAgent()
    assert masked_rule.decide(masked).action_types == ("dropout_handle",)
    assert masked_rule.decide({**masked, "round": 41}).action_types == (
        "dropout_handle",
    )
    assert masked_rule.decide({
        **masked, "round": 42, "acc_delta_3": 0.0,
    }).action_types == ("dropout_handle",)

    label_rule = CompositeDiagnoseAgent()
    label_obs = {
        **obs, "acc_delta_3": 0.0, "label_shift": 0.3,
        "participation_gap": 0.0,
    }
    assert label_rule.decide(label_obs).type == "no_op"
    assert label_rule.decide({
        **label_obs, "round": 41,
    }).action_types == ("label_prior_adapt",)

    fault_rule = CompositeDiagnoseAgent()
    fault_obs = {
        **label_obs, "label_shift": 0.0, "client_loss_mean_delta": 0.0,
        "update_norm_ratio": 2.1,
    }
    assert fault_rule.decide(fault_obs).type == "no_op"
    assert fault_rule.decide({
        **fault_obs, "round": 41,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.01,
    }).action_types == ("robust",)
    assert fault_rule.decide({
        **fault_obs, "round": 42, "update_norm_ratio": 1.4,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.01,
    }).action_types == ("robust",)
    assert fault_rule.decide({
        **fault_obs, "round": 43, "update_norm_ratio": 1.2,
    }).action_types == ("robust",)
    assert fault_rule.decide({
        **fault_obs, "round": 44, "update_norm_ratio": 1.2,
    }).type == "no_op"

    core_rule = CompositeDiagnoseAgent(core_only=True)
    real_fault_obs = {
        **fault_obs, "acc_delta_3": -0.5, "client_loss_mean_delta": 0.2,
        "label_shift": 0.10, "client_score_exceedance_fraction": 0.70,
        "update_norm_ratio": 2.1,
    }
    assert core_rule.decide(real_fault_obs).type == "no_op"
    assert core_rule.decide({
        **real_fault_obs, "round": 41,
    }).action_types == ("drift_adapt", "robust")

    pure_fault = CompositeDiagnoseAgent(core_only=True)
    pure_fault_obs = {
        **fault_obs, "acc_delta_3": -0.5, "participation_gap": 0.0,
    }
    assert pure_fault.decide(pure_fault_obs).type == "no_op"
    assert pure_fault.decide({
        **pure_fault_obs, "round": 41,
    }).action_types == ("robust",)

    fault_dropout = CompositeDiagnoseAgent(core_only=True)
    fault_dropout_obs = {**fault_obs, "participation_gap": 0.5}
    assert fault_dropout.decide(fault_dropout_obs).action_types == ("dropout_handle",)
    assert fault_dropout.decide({
        **fault_dropout_obs, "round": 41, "acc_delta_3": -0.5,
    }).action_types == ("robust", "dropout_handle")

    virtual_override = CompositeDiagnoseAgent(core_only=True)
    override_obs = {
        **pure_fault_obs, "update_norm_ratio": 1.0,
        "client_loss_mean_delta": 0.2, "label_shift": 0.10,
        "client_score_exceedance_fraction": 0.70,
    }
    assert virtual_override.decide(override_obs).type == "no_op"
    assert virtual_override.decide({
        **override_obs, "round": 41,
    }).action_types == ("drift_adapt",)
    assert virtual_override.decide({
        **override_obs, "round": 42, "input_shift": 2.0,
    }).action_types == ("moment_align_adapt",)
    assert CompositeDiagnoseAgent(core_only=True).decide(label_obs).type == "no_op"
    core_staggered_obs = {
        **label_obs, "label_shift": 0.0, "update_norm_var": 0.03,
        "update_norm_ratio": 1.5,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.20,
    }
    assert CompositeDiagnoseAgent(core_only=True).decide(
        core_staggered_obs,
    ).type == "no_op"

    staggered_rule = CompositeDiagnoseAgent()
    staggered_obs = {
        **label_obs, "label_shift": 0.0, "acc_delta_3": -0.5,
        "client_loss_mean_delta": 0.0,
        "update_norm_var": 0.03, "update_norm_ratio": 1.5,
    }
    assert staggered_rule.decide(staggered_obs).type == "no_op"
    assert staggered_rule.decide({
        **staggered_obs, "round": 41,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.20,
    }).type == "no_op"
    assert staggered_rule.decide({
        **staggered_obs, "round": 42,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.20,
    }).action_types == ("spawn_concept_auto",)

    split_evidence_rule = CompositeDiagnoseAgent()
    split_evidence_obs = {
        **staggered_obs, "round": 40,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.141,
        "client_model_score_drop_p90": 0.294,
        "online_roster_overlap": 1.0,
        "client_score_exceedance_fraction": 0.60,
        "client_score_exceedance_overlap": 0.0,
        "client_update_group_separation": 0.252,
    }
    assert split_evidence_rule.decide(split_evidence_obs).type == "no_op"
    assert split_evidence_rule.decide({
        **split_evidence_obs, "round": 41,
        "client_score_exceedance_fraction": 0.70,
        "client_score_exceedance_overlap": 0.857,
        "client_update_group_separation": -0.013,
    }).action_types == ("spawn_concept_auto",)
    assert split_evidence_rule.decide({
        **split_evidence_obs, "round": 60,
        "update_norm_ratio": 2.1,
    }).type == "no_op"
    assert split_evidence_rule.decide({
        **split_evidence_obs, "round": 61,
        "update_norm_ratio": 2.1,
    }).type == "no_op"

    virtual_spawn_guard = CompositeDiagnoseAgent()
    assert virtual_spawn_guard.decide({
        **split_evidence_obs, "input_shift": 2.0,
    }).action_types == ("moment_align_adapt",)
    assert virtual_spawn_guard.decide({
        **split_evidence_obs, "round": 41,
        "input_shift": 0.0,
        "client_score_exceedance_overlap": 0.857,
    }).action_types == ("moment_align_adapt",)

    fault_like_rule = CompositeDiagnoseAgent()
    fault_like_obs = {
        **staggered_obs, "label_shift": 0.17,
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "client_score_exceedance_fraction": 0.30,
        "client_update_group_separation": 0.01,
    }
    assert fault_like_rule.decide({
        **fault_like_obs, "round": 41, "update_norm_ratio": 1.52,
    }).type == "no_op"
    fault_second = fault_like_rule.decide({
        **fault_like_obs, "round": 42, "update_norm_ratio": 1.65,
    })
    assert fault_second.type == "no_op"
    assert fault_like_rule.decide({
        **fault_like_obs, "round": 43, "update_norm_ratio": 1.8,
    }).action_types == ("robust",)
    assert not fault_like_rule.staggered_detected
    hetero_rule = CompositeDiagnoseAgent()
    hetero_obs = {
        **obs, "round": 0, "acc_delta_3": 0.0, "label_shift": 0.0,
        "participation_rate": 0.4, "planned_participation_rate": 0.4,
        "availability_rate": 1.0, "participation_gap": 0.0,
        "select_enabled": True, "select_k": 1,
        "client_utilities": [{
            "client": 0, "utility": 1.0, "oort_score": 1.0, "oort_unexplored": False,
            "oort_blacklisted": False, "available": True,
        }],
    }
    for round_idx in range(2):
        hetero_obs.update(round=round_idx, update_norm_var=0.1)
        assert hetero_rule.decide(hetero_obs).type == "no_op"
    hetero_obs.update(round=2, update_norm_var=0.1)
    assert hetero_rule.decide(hetero_obs).type == "select_clients"
    dropout_hetero_obs = {
        **hetero_obs, "round": 40, "availability_rate": 0.5,
        "participation_gap": 0.2,
    }
    assert {"dropout_handle", "select_clients"} <= set(
        hetero_rule.decide(dropout_hetero_obs).action_types
    )
    assert {"dropout_handle", "select_clients"} <= set(hetero_rule.decide({
        **dropout_hetero_obs, "round": 41, "participation_gap": 0.0,
    }).action_types)
    assert "dropout_handle" not in hetero_rule.decide({
        **dropout_hetero_obs, "round": 42, "availability_rate": 1.0,
        "participation_gap": 0.0,
    }).action_types
    real_hetero_obs = {
        **hetero_obs, "round": 40, "acc_delta_3": -0.05,
        "label_shift": 0.10, "client_score_exceedance_fraction": 0.75,
    }
    assert hetero_rule.decide(real_hetero_obs).type == "select_clients"
    assert hetero_rule.decide({
        **real_hetero_obs, "round": 41,
    }).action_types == ("drift_adapt", "select_clients")
    rendered = render_text(obs)
    assert "Planned participation 1.00" in rendered and "participation gap 0.50" in rendered
    llm = LLMAgent(
        query_fn=lambda _: '{"actions":["drift_adapt","dropout_handle"],"subtype":"real"}',
        allowed_actions=B3_VISIBLE_ACTIONS,
    )
    parsed = llm.decide(obs)
    assert parsed.action_types == ("drift_adapt", "dropout_handle")
    conflict = LLMAgent(
        query_fn=lambda _: '{"actions":["drift_adapt","moment_align_adapt"]}',
        allowed_actions=B3_VISIBLE_ACTIONS,
    ).decide(obs)
    assert conflict.type == "no_op"
    assert LLMAgent(query_fn=lambda _: "do not use robust").decide(obs).type == "no_op"
    assert LLMAgent(query_fn=lambda _: '{"actions":["robust"],"reason":"x"}').decide(obs).type == "no_op"
    calls = []
    responses = iter(('{"actions":["select_clients"]}', '{"actions":["no_op"]}'))
    per_round_llm = LLMAgent(query_fn=lambda text: calls.append(text) or next(responses))
    assert per_round_llm.decide(hetero_obs).type == "select_clients"
    per_round_llm.record_decision_feedback(
        round_idx=2,
        declared_actions=("select_clients",),
        status="VALID",
        activated_actions=("select_clients",),
        started_actions=("select_clients",),
        stopped_actions=(),
        effective_from_round=3,
        observation=hetero_obs,
    )
    next_obs = {
        **hetero_obs,
        "round": 3,
        "active_actions": ["select_clients"],
        "global_acc": hetero_obs["global_acc"] + 0.02,
        "client_loss_mean": hetero_obs["client_loss_mean"] - 0.1,
        "update_norm_ratio": hetero_obs["update_norm_ratio"] - 0.5,
    }
    assert per_round_llm.decide(next_obs).type == "no_op"
    assert len(calls) == 2
    llm_view = json.loads(calls[0])
    assert "client_utilities" not in llm_view
    assert "input_shift" not in llm_view and "label_shift" not in llm_view
    assert llm_view["input_moment_delta"] == hetero_obs["input_shift"]
    assert llm_view["aggregate_label_hist_tv"] == hetero_obs["label_shift"]
    assert llm_view["client_change_monitor_ready"] == hetero_obs["client_change_monitor_ready"]
    assert llm_view["client_model_score_drop_p50"] == hetero_obs["client_model_score_drop_p50"]
    assert llm_view["client_model_score_drop_p90"] == hetero_obs["client_model_score_drop_p90"]
    assert llm_view["client_update_direction_dispersion"] == hetero_obs["client_update_direction_dispersion"]
    for field in (
        "online_roster_overlap", "client_score_exceedance_fraction",
        "client_score_exceedance_overlap", "client_update_group_separation",
    ):
        assert llm_view[field] == hetero_obs[field]
    assert llm_view["client_loss_mean_delta"] == hetero_obs["client_loss_mean_delta"]
    assert llm_view["utility_client_count"] == len(hetero_obs["client_utilities"])
    assert llm_view["utility_max"] == 1.0
    assert not ({
        "feddrift_ready", "feddrift_trigger", "feddrift_loss_jump",
        "local_loss_detector_ready", "local_loss_change_score",
        "select_k", "select_enabled", "select_seed", "oort_exploration",
        "oort_round_threshold", "oort_prefer_duration", "oort_sample_window",
        "utility_top_k", "blacklisted_client_count", "unexplored_client_count",
    } & set(llm_view))
    assert llm_view["last_decision"] is None
    assert llm_view["active_action_ages"] == {}
    next_llm_view = json.loads(calls[1])
    assert next_llm_view["last_decision"] == {
        "round": 2,
        "declared_actions": ["select_clients"],
        "status": "VALID",
        "activated_actions": ["select_clients"],
    }
    assert next_llm_view["active_action_ages"] == {"select_clients": 1}
    assert next_llm_view["telemetry_delta_since_last_decision"] == {
        "global_acc": 0.02,
        "client_loss_mean": -0.1,
        "update_norm_ratio": -0.5,
        "participation_gap": 0.0,
        "client_model_score_drop_p50": 0.0,
        "client_model_score_drop_p90": 0.0,
        "client_update_direction_dispersion": 0.0,
        "online_roster_overlap": 0.0,
        "client_score_exceedance_fraction": 0.0,
        "client_score_exceedance_overlap": 0.0,
        "client_update_group_separation": 0.0,
    }
    failed_memory = LLMAgent(query_fn=lambda _: '{"actions":["no_op"]}')
    failed_memory.record_decision_feedback(
        round_idx=3,
        declared_actions=("robust",),
        status="INVALID_SCHEMA",
        activated_actions=("robust",),
        started_actions=(),
        stopped_actions=(),
        effective_from_round=4,
        observation=next_obs,
    )
    failed_memory.decide({**next_obs, "round": 4})
    failed_view = json.loads(failed_memory.last_public_observation_json)
    assert failed_view["last_decision"]["status"] == "INVALID_SCHEMA"
    assert failed_view["last_decision"]["activated_actions"] == []
    oracle = CompositeOracleAgent(40, ("drift_adapt", "dropout_handle"))
    assert oracle.decide({"round": 39}).type == "no_op"
    assert oracle.decide(obs).action_types == ("drift_adapt", "dropout_handle")
    one_shot_oracle = CompositeOracleAgent(40, ("spawn_concept_auto",))
    assert one_shot_oracle.decide({"round": 39}).type == "no_op"
    assert one_shot_oracle.decide({"round": 40}).type == "spawn_concept_auto"
    assert one_shot_oracle.decide({"round": 41}).type == "no_op"
    assert "between one and three" in B3_SYSTEM_PROMPT and "Return JSON only" in B3_SYSTEM_PROMPT
    assert "lr_reset" not in B3_SYSTEM_PROMPT and "spawn_concept:" not in B3_SYSTEM_PROMPT
    assert all(action in ACTION_FAMILY for action in B3_VISIBLE_ACTIONS)

    cfg = make_cfg("dropout", 7, synthetic=True, smoke=True, decide_every=1)
    cfg.num_clients = 4
    cfg.samples_per_client = 40
    cfg.test_size = 100
    cfg.rounds = 2
    cfg.drift_round = 1
    cfg.local_epochs = 1
    set_seed(cfg.seed)
    hist = Engine(cfg, build_data(cfg)).run(BundleAgent(), log=False)
    assert hist["actions"][:2] == [(0, "dropout_handle"), (0, "robust")]
    assert hist["telemetry"][0]["action_bundle"] == "dropout_handle|robust"
    assert hist["telemetry"][0]["active_before"] == "no_op"
    assert hist["telemetry"][0]["active_after"] == "dropout_handle|robust"
    assert hist["telemetry"][0]["effective_from_round"] == 1
    assert hist["telemetry"][0]["robust_variant"] == "none"
    assert hist["telemetry"][1]["active_before"] == "dropout_handle|robust"
    assert hist["telemetry"][1]["active_after"] == "no_op"
    assert hist["telemetry"][1]["robust_variant"] == "distance_trimmed_mean"
    assert hist["tool_bundle"] == ["dropout_handle", "robust"]
    assert hist["telemetry"][1]["dropout_handle_substitutions"] == 2

    empty_cfg = make_cfg("dropout", 9, synthetic=True, smoke=True, decide_every=1)
    empty_cfg.num_clients = 4
    empty_cfg.samples_per_client = 20
    empty_cfg.test_size = 50
    empty_cfg.rounds = 1
    empty_cfg.drift_round = 0
    empty_data = build_data(empty_cfg)
    empty_data["dropped_clients"] = set(range(empty_cfg.num_clients))
    empty_hist = Engine(empty_cfg, empty_data).run(BundleAgent(), log=False)
    empty_row = empty_hist["telemetry"][0]
    assert empty_row["empty_round"] and empty_row["participation_rate"] == 0.0
    assert empty_row["update_norm_mean"] == 0.0
    assert empty_row["dropout_handle_substitutions"] == 0

    set_seed(cfg.seed)
    stale_hist = Engine(cfg, build_data(cfg)).run(StaleAgent(), log=False)
    sources = {row["source"]: row["count"]
               for row in stale_hist["telemetry"][1]["aggregation_sources"]}
    assert sources == {"online_real": 2, "stale_cache": 2}
    assert norm_stats([1.0, 1.0])["ratio"] == 1.0
    assert norm_stats([1.0, 1.0, 10.0])["ratio"] == 10.0

    poison_cfg = make_cfg("fault", 21, synthetic=True, smoke=True, decide_every=1)
    poison_cfg.num_clients = 4
    poison_cfg.samples_per_client = 40
    poison_cfg.test_size = 100
    poison_cfg.rounds = 3
    poison_cfg.drift_round = 1
    poison_cfg.local_epochs = 1
    poison_cfg.byzantine_frac = 0.25
    poison_data = build_data(poison_cfg)
    poison_data["client_X"][0] = poison_data["client_X"][0] + 20.0
    poison_data["b3_episode_plan"] = {
        "episode_id": "ep_" + "2" * 32,
        "plan_sha256": "poison_fixture",
        "partition_sha256": poison_data["partition_sha256"],
        "hidden_event": {
            "active_causes": ["fault", "dropout"],
            "affected_clients": {"fault": {"clients": [0]}, "dropout": {"clients": [0]}},
            "event_schedule": [{"round": 1, "activate": ["fault", "dropout"], "severity": {}}],
            "cause_intervals": {
                "fault": {"0": [[1, 2]]},
                "dropout": {"0": [[2, None]]},
            },
        },
    }
    set_seed(poison_cfg.seed)
    poison_hist = Engine(poison_cfg, poison_data).run(
        PersistentBundleAgent("robust", "dropout_handle"), log=False
    )
    assert poison_hist["telemetry"][1]["update_norm_ratio"] > 2.0
    assert "0" not in poison_hist["telemetry"][1]["robust_selected_clients"].split()
    assert poison_hist["telemetry"][1]["cache_rejection_count"] >= 1
    poison_sources = {row["source"] for row in poison_hist["telemetry"][2]["aggregation_sources"]}
    assert "stale_cache" not in poison_sources

    feddaa_cfg = make_cfg("drift", 22, synthetic=True, smoke=True, decide_every=1)
    feddaa_cfg.num_clients = 4
    feddaa_cfg.samples_per_client = 24
    feddaa_cfg.test_size = 50
    feddaa_cfg.rounds = 5
    feddaa_cfg.drift_round = 0
    feddaa_cfg.local_epochs = 1
    feddaa_cfg.feddaa_T = 3
    feddaa_data = build_data(feddaa_cfg)
    feddaa_data["b3_episode_plan"] = {
        "episode_id": "ep_" + "3" * 32, "plan_sha256": "feddaa_fixture",
        "partition_sha256": feddaa_data["partition_sha256"],
        "hidden_event": {
            "active_causes": ["real_drift", "dropout"],
            "affected_clients": {
                "real_drift": {"clients": [0, 1, 2, 3]}, "dropout": {"clients": [0]},
            },
            "event_schedule": [{"round": 0, "activate": ["real_drift", "dropout"], "severity": {}}],
            "cause_intervals": {
                "real_drift": {str(ci): [[0, None]] for ci in range(4)},
                "dropout": {"0": [[1, None]]},
            },
        },
    }
    proto_clients = []
    original_proto = engine_module._model_proto
    client_by_tensor = {id(value): ci for ci, value in enumerate(feddaa_data["client_X"])}
    def recording_proto(model, values, labels, n_classes, device):
        proto_clients.append(client_by_tensor.get(id(values)))
        return original_proto(model, values, labels, n_classes, device)
    engine_module._model_proto = recording_proto
    try:
        set_seed(feddaa_cfg.seed)
        feddaa_hist = Engine(feddaa_cfg, feddaa_data).run(
            PersistentBundleAgent("drift_adapt", "dropout_handle"), log=False
        )
    finally:
        engine_module._model_proto = original_proto
    assert proto_clients and 0 not in proto_clients
    rebuilds = [(row["round"], row["feddaa_event"])
                for row in feddaa_hist["telemetry"] if row["feddaa_event"] != "none"]
    assert rebuilds == [(1, "init"), (4, "prototype_silhouette")], rebuilds
    assert all(row["feddaa_report_clients"] == 3
               for row in feddaa_hist["telemetry"][1:])
    feddaa_summary = summarize(feddaa_hist, feddaa_cfg, 0)
    assert feddaa_summary["feddaa_rebuild_count"] == 2
    assert feddaa_summary["feddaa_report_bytes"] > 0
    assert feddaa_summary["feddaa_model_evaluations"] > 0

    feddrift_cfg = make_cfg("staggered", 23, synthetic=True, smoke=True, decide_every=1)
    feddrift_cfg.num_clients = 4
    feddrift_cfg.samples_per_client = 24
    feddrift_cfg.test_size = 50
    feddrift_cfg.rounds = 2
    feddrift_cfg.drift_round = 1
    feddrift_cfg.local_epochs = 1
    feddrift_data = build_data(feddrift_cfg)
    feddrift_data["b3_episode_plan"] = {
        "episode_id": "ep_" + "4" * 32, "plan_sha256": "feddrift_fixture",
        "partition_sha256": feddrift_data["partition_sha256"],
        "hidden_event": {
            "active_causes": ["dropout"],
            "affected_clients": {"dropout": {"clients": [0]}},
            "event_schedule": [{"round": 1, "activate": ["dropout"], "severity": {}}],
            "cause_intervals": {"dropout": {"0": [[1, None]]}},
        },
    }
    acc_clients = []
    original_acc = engine_module._model_acc
    client_by_tensor = {id(value): ci for ci, value in enumerate(feddrift_data["client_X"])}
    def recording_acc(model, values, labels, device, max_samples=0):
        acc_clients.append(client_by_tensor.get(id(values)))
        return original_acc(model, values, labels, device, max_samples)
    engine_module._model_acc = recording_acc
    try:
        set_seed(feddrift_cfg.seed)
        feddrift_hist = Engine(feddrift_cfg, feddrift_data).run(
            PersistentBundleAgent("spawn_concept_auto"), log=False
        )
    finally:
        engine_module._model_acc = original_acc
    assert acc_clients.count(0) == 1
    assert all(acc_clients.count(ci) > 1 for ci in (1, 2, 3))
    assert feddrift_hist["telemetry"][0]["feddrift_sc_weight_bins"] == 0
    assert feddrift_hist["telemetry"][0]["feddrift_assignment_records"] == 0
    assert feddrift_hist["telemetry"][1]["feddrift_assignment_records"] == 3

    selector_obs = collect(0, 0.5, [0.5], [1.0], [1.0], 0.1, 0.5, False, 2)
    selector_obs.update({
        "select_enabled": True, "select_k": 1, "oort_exploration": 0.0,
        "client_utilities": [
            {"client": 0, "oort_score": 100.0, "oort_unexplored": False,
             "oort_blacklisted": False, "available": False},
            {"client": 1, "oort_score": 1.0, "oort_unexplored": False,
             "oort_blacklisted": False, "available": True},
        ],
    })
    assert SelectClientsAgent().decide(selector_obs).clients == (1,)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "schema_v1.json"
        episode_id = "ep_" + "0" * 32
        checkpoint.write_text(json.dumps({
            "schema_version": 1,
            "episode_id": episode_id,
            "agent": "noop",
            "implementation_sha256": "fixture",
            "history": {},
        }), encoding="utf-8")
        try:
            _checkpoint_history(
                checkpoint, SimpleNamespace(name="noop"), SimpleNamespace(seed=0),
                {"b3_episode_plan": {"schema_version": 2}}, episode_id,
                "fixture", True,
            )
        except ValueError as exc:
            assert "version/fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("schema-v1 checkpoint did not fail closed")
        checkpoint.write_text(json.dumps({
            "schema_version": 2,
            "episode_id": episode_id,
            "agent": "noop",
            "run_fingerprint_sha256": "old",
            "history": {},
        }), encoding="utf-8")
        try:
            _checkpoint_history(
                checkpoint, SimpleNamespace(name="noop"), SimpleNamespace(seed=0),
                {"b3_episode_plan": {"schema_version": 2}}, episode_id,
                "new", True,
            )
        except ValueError as exc:
            assert "version/fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("stale run fingerprint did not fail closed")

        class FailingEngine:
            def __init__(self, cfg, data):
                self.current_round = 7

            def run(self, agent, log=False):
                raise RuntimeError("fixture execution failure")

        failure_checkpoint = Path(tmp) / "execution_failure.json"
        original_engine = composite_module.Engine
        composite_module.Engine = FailingEngine
        try:
            try:
                _checkpoint_history(
                    failure_checkpoint, SimpleNamespace(name="fixture_agent"),
                    SimpleNamespace(seed=0, decide_every=1),
                    {"b3_episode_plan": {"schema_version": 2}}, episode_id,
                    "fixture", False, audit_run_id="run_fixture",
                )
            except RuntimeError as exc:
                assert "fixture execution failure" in str(exc)
            else:
                raise AssertionError("benchmark execution failure was swallowed")
        finally:
            composite_module.Engine = original_engine
        failure = json.loads(
            failure_checkpoint.with_suffix(".run_integrity.json").read_text(
                encoding="utf-8"
            )
        )["integrity"]
        assert failure["failure_type"] == "BENCHMARK_EXECUTION_FAILED"
        assert failure["failure_round"] == 7
        assert failure["rerun_required"] is True
    print("B3_PHASE2_OK")


if __name__ == "__main__":
    main()
