"""Logic smoke test (no matplotlib). Verifies engine + 4 actions + scoring on
synthetic CPU data for all three scenarios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from config import Config
from data import load_data, dirichlet_partition, make_drift_map
from engine import (Engine, _aggregate_label_reports, _simulated_secure_input_shift,
                    _simulated_secure_label_shift,
                    _distance_trimmed_indices,
                    _fdms_pairwise_similarity, _fdms_update_matrix,
                    _init_oort_state, _oort_update_client, _oort_scores,
                    _feddrift_cluster_clients, _feddrift_cluster_report,
                    _feddrift_match_models, _feddrift_stable_groups, _feddrift_update_detector,
                    _feddrift_update_reports, _feddaa_auto_assignment,
                    _feddaa_rebuild_due, _feddaa_recluster, _softmax_neg)
from agents import (NoOpAgent, AdaptAgent, RejectAgent, HandleDropoutAgent,
                    FriendSubstituteAgent, SelectClientsAgent, AutoSpawnConceptAgent, DiagnoseAgent,
                    OracleAgent, LLMAgent)
from scoring import correct_tools_for, summarize
from run import build_data
from telemetry import clustering_metrics, collect
from runtime_contract import (FedDriftReport, InputShiftReport,
                              LabelHistogramReport, LabelShiftReport)
from observation_contract import PUBLIC_OBSERVATION_SCHEMA


def build(cfg):
    np.random.seed(0); torch.manual_seed(0)
    (Xtr, ytr), (Xte, yte), nc, ish = load_data(cfg)
    part_beta = cfg.dropout_beta if cfg.scenario == "dropout" else cfg.dirichlet_beta
    parts = dirichlet_partition(ytr, cfg.num_clients, part_beta, cfg.samples_per_client, 0)
    cy = [ytr[p] for p in parts]
    n_drift = int(cfg.drift_fraction * cfg.num_clients)
    dclients = set(np.random.default_rng(7).choice(cfg.num_clients, n_drift, replace=False).tolist()) if n_drift else set()
    dropped = set()
    if cfg.scenario == "dropout":
        half = nc // 2
        for ci, yc in enumerate(cy):
            if int(torch.bincount(yc, minlength=nc).argmax()) < half:
                dropped.add(ci)
    client_X = [Xtr[p] for p in parts]
    hists = []
    for y in cy:
        h = torch.bincount(y, minlength=nc).float()
        hists.append(h / h.sum().clamp_min(1))
    base_hist = torch.bincount(torch.cat(cy), minlength=nc).float()
    base_hist = base_hist / base_hist.sum().clamp_min(1)
    return {"client_X": client_X, "client_y": cy, "Xte": Xte, "yte": yte,
            "drift_map": make_drift_map(nc, 0), "drift_clients": dclients,
            "dropped_clients": dropped, "n_classes": nc, "input_shape": ish,
            "client_label_hist": hists,
            "baseline_label_hist": base_hist,
            "baseline_xmean": float(torch.cat(client_X).float().mean()),
            "baseline_xstd": max(float(torch.cat(client_X).float().std(unbiased=False)), 1e-6),
            "vshift_xmean": float(torch.cat(client_X).float().mean()),
            "vshift_xstd": max(float(torch.cat(client_X).float().std(unbiased=False)), 1e-6),
            "label_class_weight": torch.ones(nc),
            "label_calibration": torch.zeros(nc)}


def run(scenario):
    cfg = Config()
    cfg.scenario = scenario
    cfg.dataset = "synthetic"
    cfg.num_clients = 8; cfg.samples_per_client = 200; cfg.test_size = 500
    cfg.rounds = 20; cfg.drift_round = 10; cfg.device = "cpu"
    cfg.drift_fraction = {"drift": 1.0, "fault": 0.3, "dropout": 0.0}[scenario]
    data = build(cfg)
    eng = Engine(cfg, data)
    print(f"--- scenario={scenario}  dropped={len(data['dropped_clients'])} ---")
    for ag in [NoOpAgent(), AdaptAgent(), RejectAgent(), HandleDropoutAgent(),
               FriendSubstituteAgent(),
               OracleAgent(scenario, cfg.drift_round), LLMAgent()]:
        np.random.seed(0); torch.manual_seed(0)
        h = eng.run(ag, log=False)
        if scenario == "dropout":
            assert all(row["label_shift"] == 0.0 for row in h["telemetry"]), (
                ag.name, [row["label_shift"] for row in h["telemetry"]]
            )
        s = summarize(h, cfg, cfg.drift_round)
        print(f"  {ag.name:16s} final={s['final_acc']:.3f} post_mean={s['post_drift_mean_acc']} "
              f"tool={s['tool']}({s['tool_matched']})")


def check_dropout_window():
    """present_clients: temporary outage window drops a cluster during
    [drift_round, recover_round) and returns everyone after recover_round."""
    cfg = Config()
    cfg.scenario = "dropout"; cfg.drift_round = 10; cfg.dropout_recover_round = 15
    data = {"client_X": [None] * 8, "dropped_clients": {0, 1, 2}}
    eng = Engine.__new__(Engine); eng.cfg = cfg; eng.data = data
    assert len(eng.present_clients(5)) == 8, "before window: all present"
    assert set(eng.present_clients(12)) == {3, 4, 5, 6, 7}, "in window: cluster absent"
    assert len(eng.present_clients(15)) == 8, "at recover_round: all back"
    assert len(eng.present_clients(18)) == 8, "after recovery: all back"
    # recover_round=0 keeps the original permanent-absence behavior
    cfg.dropout_recover_round = 0
    assert len(eng.present_clients(18)) == 5, "no recovery: still absent"
    print("DROPOUT_WINDOW_OK")


def check_fdms_friend_discovery():
    gstate = {"w": torch.zeros(2)}
    states = {
        0: ({"w": torch.tensor([1.0, 0.0])}, 1),
        1: ({"w": torch.tensor([0.9, 0.1])}, 1),
        2: ({"w": torch.tensor([-1.0, 0.0])}, 1),
    }
    M = np.eye(3, dtype=float)
    T = np.zeros((3, 3), dtype=float)
    _fdms_update_matrix(M, T, _fdms_pairwise_similarity(states, gstate))
    eng = Engine.__new__(Engine); eng.data = {"client_X": [None] * 3}
    friend, sim, count = eng.friend_for(0, [1, 2], M, True)
    assert friend == 1, (friend, sim, M)
    assert sim > M[0, 2], (sim, M[0, 2])
    assert count == 2, count
    print("FDMS_FRIEND_DISCOVERY_OK")


def check_fdms_clustered_schedule():
    cfg = Config()
    cfg.scenario = "dropout"; cfg.dropout_mode = "fdms_clustered"
    cfg.num_clients = 20; cfg.dropout_rate = 0.5; cfg.drift_round = 10
    groups = [list(range(i, i + 4)) for i in range(0, 20, 4)]
    data = {"client_X": [None] * 20, "fdms_groups": groups}
    eng = Engine.__new__(Engine); eng.cfg = cfg; eng.data = data
    assert len(eng.present_clients(5)) == 20, "before dropout: all present"
    present = set(eng.present_clients(12))
    assert len(present) == 10, present
    assert all(present.intersection(g) for g in groups), present
    eng2 = Engine.__new__(Engine); eng2.cfg = cfg; eng2.data = data
    assert eng.present_clients(12) == eng2.present_clients(12), "schedule not deterministic"
    print("FDMS_CLUSTERED_SCHEDULE_OK")


def check_drift_subtype_actions():
    expected = {"real": "drift_adapt", "virtual": "moment_align_adapt", "label": "label_prior_adapt"}
    for dtype, tool in expected.items():
        cfg = Config()
        cfg.scenario = "drift"; cfg.drift_type = dtype; cfg.dataset = "synthetic"
        cfg.num_clients = 6; cfg.samples_per_client = 200; cfg.test_size = 500
        cfg.rounds = 20; cfg.drift_round = 10; cfg.device = "cpu"
        cfg.drift_fraction = 1.0
        cfg.formal_evidence = True
        np.random.seed(0); torch.manual_seed(0)
        eng = Engine(cfg, build_data(cfg))
        ag = OracleAgent("drift", cfg.drift_round, dtype)
        h = eng.run(ag, log=False)
        s = summarize(h, cfg, cfg.drift_round)
        assert s["tool"] == tool, f"{dtype}: expected {tool}, got {s['tool']}"
        assert s["tool_matched"] == "yes", f"{dtype}: tool not matched ({s})"
        assert s["evidence_status"] == "source_aligned", s
        assert "lr_reset" not in s["formal_correct_tool"], f"{dtype}: lr_reset still formal correct"
    print("DRIFT_SUBTYPE_ACTIONS_OK")


def check_moment_align_metrics():
    cfg = Config()
    cfg.scenario = "drift"; cfg.drift_type = "virtual"; cfg.dataset = "synthetic"
    cfg.num_clients = 6; cfg.samples_per_client = 200; cfg.test_size = 500
    cfg.rounds = 20; cfg.drift_round = 10; cfg.device = "cpu"
    cfg.drift_fraction = 1.0; cfg.formal_evidence = True
    np.random.seed(0); torch.manual_seed(0)
    eng = Engine(cfg, build_data(cfg))
    h = eng.run(OracleAgent("drift", cfg.drift_round, "virtual"), log=False)
    s = summarize(h, cfg, cfg.drift_round)
    assert s["tool"] == "moment_align_adapt", s
    assert s["tool_matched"] == "yes", s
    assert s["evidence_papers"] == "deep_coral", s
    assert any(row.get("moment_align_enabled") for row in h["telemetry"]), "telemetry missing moment align"
    assert not any(row.get("feddaa_enabled") for row in h["telemetry"]), "moment align should not start FedDAA"

    noisy_pre_event = {**h["telemetry"][cfg.drift_round - 1],
                       "acc_delta_3": -0.056, "input_shift": 0.0}
    virtual_event = h["telemetry"][cfg.drift_round]
    diagnose = DiagnoseAgent()
    assert diagnose.decide(noisy_pre_event).type == "no_op"
    assert diagnose.decide(virtual_event).type == "moment_align_adapt"
    llm = LLMAgent()
    assert llm.decide({key: noisy_pre_event[key] for key in PUBLIC_OBSERVATION_SCHEMA}).type == "no_op"
    assert llm.decide({key: virtual_event[key] for key in PUBLIC_OBSERVATION_SCHEMA}).type == "moment_align_adapt"
    print("MOMENT_ALIGN_METRICS_OK")


def check_label_prior_metrics():
    cfg = Config()
    cfg.scenario = "drift"; cfg.drift_type = "label"; cfg.dataset = "synthetic"
    cfg.num_clients = 6; cfg.samples_per_client = 200; cfg.test_size = 500
    cfg.rounds = 20; cfg.drift_round = 10; cfg.device = "cpu"
    cfg.drift_fraction = 1.0; cfg.formal_evidence = True
    np.random.seed(0); torch.manual_seed(0)
    data = build_data(cfg)
    assert data["label_Xte"] is not None and data["label_yte"] is not None
    train_counts = torch.bincount(torch.cat(data["label_y"]), minlength=data["n_classes"])
    test_counts = torch.bincount(data["label_yte"], minlength=data["n_classes"])
    assert bool((train_counts > 0).all()), train_counts
    assert bool((test_counts > 0).all()), test_counts
    tv = 0.5 * torch.abs(data["label_target_hist"] - data["baseline_label_hist"]).sum().item()
    assert tv > 0.1, tv
    data["label_prior_calibration"] = object()
    data["label_calibration"] = object()
    eng = Engine(cfg, data)
    h = eng.run(OracleAgent("drift", cfg.drift_round, "label"), log=False)
    s = summarize(h, cfg, cfg.drift_round)
    assert s["tool"] == "label_prior_adapt", s
    assert s["tool_matched"] == "yes", s
    assert s["evidence_status"] == "source_aligned", s
    assert s["evidence_papers"] == "logit_adjust;balms", s
    assert any(row.get("using_label_prior_adapt") for row in h["telemetry"]), "telemetry missing label prior adapt"
    active = [row for row in h["telemetry"] if row.get("using_label_prior_adapt")]
    assert active and all(row["label_prior_source"] == "online_client_histogram"
                          and row["label_prior_report_clients"] == cfg.num_clients
                          for row in active)
    assert not any(row.get("fedlc_enabled") for row in h["telemetry"]), "prior adapt should not start FedLC"
    assert any(float(row.get("label_shift", 0)) > 0 for row in h["telemetry"][cfg.drift_round:]), "label_shift missing"
    assert not any(row.get("feddaa_enabled") for row in h["telemetry"]), "label prior adapt should not start FedDAA"
    print("LABEL_PRIOR_METRICS_OK")


def check_select_clients():
    ag = SelectClientsAgent()
    obs = {
        "round": 0,
        "acc_delta_3": 0.0,
        "input_shift": 0.0,
        "label_shift": 0.0,
        "update_norm_var": 0.01,
        "select_enabled": True,
        "select_k": 2,
        "client_utilities": [
            {"client": 0, "oort_score": 0.0, "oort_reward": 1.0, "oort_unexplored": False},
            {"client": 1, "oort_score": 3.0, "oort_reward": 3.0, "oort_unexplored": False},
            {"client": 2, "oort_score": 2.0, "oort_reward": 2.0, "oort_unexplored": False},
        ],
        "select_seed": 1,
        "oort_exploration": 0.0,
        "oort_sample_window": 5,
        "participation_rate": 0.5,
    }
    action = ag.decide(obs)
    assert action.type == "select_clients", action
    assert len(action.clients) == 2 and 1 in action.clients, action

    oracle_action = OracleAgent("hetero", 0).decide(obs)
    assert oracle_action.type == "select_clients" and oracle_action.clients, oracle_action

    diagnose = DiagnoseAgent()
    for r in range(3):
        diagnose_action = diagnose.decide({**obs, "round": r, "select_seed": r})
    assert diagnose_action.type == "select_clients" and diagnose_action.clients, diagnose_action
    assert diagnose.decide({**obs, "round": 10}).type == "select_clients"

    diagnose = DiagnoseAgent()
    for r in (0, 10, 20):
        diagnose_action = diagnose.decide({**obs, "round": r, "decision_interval": 10,
                                           "select_seed": r})
    assert diagnose_action.type == "select_clients" and diagnose_action.clients, diagnose_action

    llm_obs = collect(0, 0.3, [0.3], [1.0], [0.5], 0.1, 0.5, False, 20)
    llm_obs.update(obs)
    llm = LLMAgent(query_fn=lambda _: '{"actions":["select_clients"]}')
    llm_action = llm.decide(llm_obs)
    assert llm_action.type == "select_clients" and llm_action.clients, llm_action
    assert llm.decide({**llm_obs, "round": 10}).type == "select_clients"

    cfg = Config()
    cfg.participation = 0.5
    data = {"client_X": [None] * 6, "dropped_clients": set()}
    eng = Engine.__new__(Engine); eng.cfg = cfg; eng.data = data
    assert eng.present_clients(0, (4, 99, 4, "x", 2)) == [4, 2]
    assert eng.last_select_invalid == 3
    assert eng.last_select_used is True
    print("SELECT_CLIENTS_OK")


def check_llm_error_fallback():
    ag = LLMAgent(query_fn=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    action = ag.decide(collect(
        0, 0.0, [0.0], [0.0], [0.0], 0.0, 1.0, False, 1,
    ))
    assert action.type == "no_op", action
    assert "agent error" in ag.last_reasoning

    def measured_query(_):
        measured_query.last_usage = {"prompt_tokens": 10, "completion_tokens": 4}
        measured_query.last_cost_usd = 0.002
        return '{"actions":["no_op"]}'
    ag = LLMAgent(query_fn=measured_query)
    obs = collect(0, 0.5, [0.5], [1.0, 1.1], [1.0, 1.1], 0.01, 1.0, False, 20)
    assert ag.decide(obs).type == "no_op"
    assert ag.called_this_round and ag.last_prompt_tokens == 10
    assert ag.last_completion_tokens == 4 and ag.last_cost_usd == 0.002
    print("LLM_ERROR_FALLBACK_OK")


def check_gradual_eval_mixture():
    cfg = Config(); cfg.scenario = "drift"; cfg.drift_mode = "gradual"
    cfg.drift_round = 2; cfg.drift_type = "real"
    eng = Engine.__new__(Engine); eng.cfg = cfg
    y = torch.tensor([0, 1])
    eng.data = {"Xte": torch.zeros(2, 1), "yte": y, "n_classes": 2,
                "drift_map": torch.tensor([1, 0]),
                "drift_schedule": {0: 2, 1: 4, 2: 6, 3: 8}}
    eng._predict = lambda model, X=None: y
    assert eng.evaluate(None, 1)[0] == 1.0
    assert eng.evaluate(None, 2)[0] == 0.75
    assert eng.evaluate(None, 4)[0] == 0.5
    assert eng.evaluate(None, 8)[0] == 0.0
    cfg.formal_evidence = True; cfg.drift_wave_frac = .25; cfg.drift_wave_every = 2
    hist = {"acc": [.5] * 10, "worst_acc": [.4] * 10, "switch_round": 3,
            "tool": "drift_adapt", "actions": [(3, "drift_adapt")],
            "telemetry": [{"round": r, "decision_due": True,
                           "action": "drift_adapt" if r == 3 else "no_op"}
                          for r in range(10)]}
    assert summarize(hist, cfg, 2)["switch_regret"] == 0
    print("GRADUAL_EVAL_MIXTURE_OK")


def check_oort_supported_selector():
    cfg = Config()
    cfg.seed = 0; cfg.oort_round_threshold = 50.0
    sizes = np.asarray([100.0, 100.0])
    state = _init_oort_state(sizes, cfg)
    state["duration"][:] = [50.0, 200.0]
    for ci in (0, 1):
        _oort_update_client(state, ci, 10.0, 0)
    scores, _ = _oort_scores(state, sizes, np.ones(2), np.ones(2), 1, cfg)
    assert scores[0]["oort_score"] > scores[1]["oort_score"], scores

    state["duration"][:] = [50.0, 50.0]
    state["last"][:] = [4, 0]
    scores, _ = _oort_scores(state, sizes, np.ones(2), np.ones(2), 5, cfg)
    assert scores[1]["oort_uncertainty"] > scores[0]["oort_uncertainty"], scores

    cfg.scenario = "hetero"; cfg.formal_evidence = True; cfg.participation = 1.0
    assert "select_clients" not in correct_tools_for(cfg)
    cfg.participation = 0.5
    assert correct_tools_for(cfg) == ("select_clients",), correct_tools_for(cfg)

    print("OORT_SUPPORTED_SELECTOR_OK")


def check_distance_trimmed_mean():
    states = [{"w": torch.tensor([x], dtype=torch.float32)}
              for x in [0, 1, 2, 100, 101]]
    chosen, actual = _distance_trimmed_indices(states, 1)
    assert chosen == [0, 1, 2], (chosen, actual)
    assert actual == 1
    chosen, actual = _distance_trimmed_indices(states, 2)
    assert len(chosen) == 1, (chosen, actual)
    assert actual == 2
    print("DISTANCE_TRIMMED_MEAN_OK")


def check_relative_fault_trigger():
    obs = collect(1, 0.5, [0.5], [1.0, 1.0, 1.0], [1.0, 1.0, 4.0],
                  0.1, 1.0, False, 10)
    assert obs["update_norm_max"] == 4.0
    assert obs["update_norm_median"] == 1.0
    assert obs["update_norm_ratio"] == 4.0

    ag = RejectAgent()
    base = {"using_robust": False}
    assert ag.decide({**base, "update_norm_mean": 1.0, "update_norm_ratio": 1.1}).type == "no_op"
    assert ag.decide({**base, "update_norm_mean": 0.8, "update_norm_ratio": 2.5}).type == "no_op"
    assert ag.decide({**base, "update_norm_mean": 1.0, "update_norm_ratio": 2.2}).type == "robust"
    assert ag.decide({**base, "update_norm_ratio": 1.4}).type == "robust"
    assert ag.decide({**base, "update_norm_ratio": 1.2}).type == "robust"
    assert ag.decide({**base, "update_norm_ratio": 1.2}).type == "no_op"

    sparse_votes = RejectAgent()
    assert sparse_votes.decide({**base, "update_norm_ratio": 1.6}).type == "no_op"
    assert sparse_votes.decide({**base, "update_norm_ratio": 1.5}).type == "no_op"
    assert sparse_votes.decide({**base, "update_norm_ratio": 1.58}).type == "no_op"
    assert sparse_votes.decide({**base, "update_norm_ratio": 1.77}).type == "robust"

    for explained in ({"input_shift": 2.0}, {"label_shift": 0.3}, {"participation_rate": 0.5}):
        ag = RejectAgent()
        evidence = {**base, **explained, "update_norm_mean": 1.5, "update_norm_ratio": 2.5}
        assert ag.decide(evidence).type == "no_op"
        action = ag.decide(evidence)
        assert action.type == "robust", (explained, action)
    print("RELATIVE_FAULT_TRIGGER_OK")


def check_feddrift_supported_primitives():
    cfg = Config()
    cfg.feddrift_cluster_threshold = 0.2
    cfg.feddrift_min_cluster_size = 2
    features = {
        0: np.asarray([1.0, 0.0]),
        1: np.asarray([0.9, 0.1]),
        2: np.asarray([-1.0, 0.0]),
        3: np.asarray([-0.9, -0.1]),
    }
    assignment, split = _feddrift_cluster_clients(features, cfg)
    assert len(set(assignment.values())) == 2, (assignment, split)
    truth_a = {0: 1, 1: 1, 2: 2, 3: 2}
    truth_b = {0: 2, 1: 2, 2: 1, 3: 1}
    assert _feddrift_cluster_report(assignment, truth_a) == 1.0
    assert assignment == _feddrift_cluster_clients(features, cfg)[0]
    assert _feddrift_cluster_report(assignment, truth_b) == 1.0

    auto = AutoSpawnConceptAgent()
    assert auto.decide({"client_change_monitor_ready": False}).type == "no_op"
    staggered_signal = {
        "client_change_monitor_ready": True,
        "client_model_score_drop_p50": 0.05,
        "client_model_score_drop_p90": 0.20,
        "client_update_direction_dispersion": 0.30,
        "online_roster_overlap": 0.90,
        "client_score_exceedance_fraction": 0.30,
        "client_score_exceedance_overlap": 0.70,
        "client_update_group_separation": 0.20,
        "planned_participation_rate": 1.0,
    }
    assert auto.decide({
        **staggered_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.5,
    }).type == "no_op"
    assert auto.decide({
        **staggered_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.5,
    }).type == "spawn_concept_auto"
    assert auto.decide({
        **staggered_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.5,
    }).type == "no_op"

    below_floor = AutoSpawnConceptAgent()
    below_floor_signal = {
        **staggered_signal,
        "client_model_score_drop_p50": 0.01,
        "client_model_score_drop_p90": 0.15,
    }
    assert below_floor.decide(below_floor_signal).type == "no_op"
    assert below_floor.decide(below_floor_signal).type == "no_op"

    dropout_guard = AutoSpawnConceptAgent()
    dropout_signal = {**staggered_signal, "participation_gap": 0.5}
    assert dropout_guard.decide(dropout_signal).type == "no_op"
    assert dropout_guard.decide(dropout_signal).type == "no_op"

    virtual_guard = AutoSpawnConceptAgent()
    assert virtual_guard.decide({
        **staggered_signal, "input_shift": 2.0,
    }).type == "no_op"
    assert virtual_guard.decide({
        **staggered_signal, "input_shift": 0.0,
    }).type == "no_op"

    transient_guard = AutoSpawnConceptAgent()
    no_direction = {
        **staggered_signal, "client_update_group_separation": -0.10,
    }
    assert transient_guard.decide(no_direction).type == "no_op"
    assert transient_guard.decide(no_direction).type == "no_op"
    assert transient_guard.decide(staggered_signal).type == "spawn_concept_auto"
    assert transient_guard.decide(staggered_signal).type == "no_op"

    fault_like = AutoSpawnConceptAgent()
    fault_signal = {
        **staggered_signal, "client_update_group_separation": 0.01,
    }
    assert fault_like.decide({
        **fault_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.52,
    }).type == "no_op"
    assert fault_like.decide({
        **fault_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.65,
    }).type == "no_op"
    assert fault_like.decide({
        **fault_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 1.2,
    }).type == "no_op"

    delayed_guard = AutoSpawnConceptAgent()
    assert delayed_guard.decide({
        **staggered_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 2.0,
    }).type == "no_op"
    assert delayed_guard.decide({
        **staggered_signal, "update_norm_var": 0.03,
        "update_norm_ratio": 2.0,
    }).type == "spawn_concept_auto"
    assert delayed_guard.decide({
        **staggered_signal,
        "update_norm_var": 0.03,
        "update_norm_ratio": 2.0,
    }).type == "no_op"

    ref = np.full(1, np.nan); ema = np.zeros(1); confirm = np.zeros(1, dtype=int)
    for loss in [1.0, 0.9, 1.1]:
        _feddrift_update_detector(ref, ema, confirm, 0, loss, cfg)
    assert confirm[0] < cfg.feddrift_confirm_rounds
    _feddrift_update_detector(ref, ema, confirm, 0, 1.2, cfg)
    _feddrift_update_detector(ref, ema, confirm, 0, 1.2, cfg)
    assert confirm[0] >= cfg.feddrift_confirm_rounds

    cfg.feddrift_warmup_rounds = 0
    cfg.feddrift_confirm_rounds = 2
    ref_acc = np.full(3, 0.9); confirmations = np.zeros(3, dtype=int)
    report_scores = np.zeros(3, dtype=float)
    first = [FedDriftReport(ci, 5, 10, ((1, 0, 0.5),)) for ci in (0, 1)]
    before_confirm = confirmations.copy()
    try:
        _feddrift_update_reports(first + first[:1], ref_acc, confirmations,
                                 report_scores, cfg)
        raise AssertionError("duplicate FedDrift report accepted")
    except ValueError:
        pass
    assert np.array_equal(confirmations, before_confirm)
    confirmed, _ = _feddrift_update_reports(
        first, ref_acc, confirmations, report_scores, cfg
    )
    assert confirmed == [] and list(confirmations[:2]) == [1, 1]
    second = [FedDriftReport(ci, 6, 10, ((1, 0, 0.5),)) for ci in (0, 1)]
    confirmed, _ = _feddrift_update_reports(
        second, ref_acc, confirmations, report_scores, cfg
    )
    stable = _feddrift_stable_groups(confirmed, features, cfg)
    assert stable == [[0, 1]], stable
    assert _feddrift_stable_groups([0], features, cfg) == []

    recurrent = FedDriftReport(0, 7, 10, ((1, 4, 0.92), (2, 2, 0.31)))
    assert _feddrift_match_models([recurrent]) == {0: 1}

    perfect = clustering_metrics(
        {0: 5, 1: 5, 2: 8, 3: 8}, {0: 0, 1: 0, 2: 1, 3: 1}
    )
    assert perfect == {"purity": 1.0, "ari": 1.0, "nmi": 1.0,
                       "learner_count_error": 0}
    oversplit = clustering_metrics(
        {0: 0, 1: 1, 2: 2, 3: 3}, {0: 0, 1: 0, 2: 1, 3: 1}
    )
    assert oversplit["purity"] == 1.0
    assert oversplit["learner_count_error"] == 2 and oversplit["ari"] < 1.0

    evaluator = Engine.__new__(Engine)
    evaluator.cfg = Config(); evaluator.cfg.n_concepts = 2
    evaluator.data = {
        "yte": torch.tensor([0, 1]),
        "drift_maps": {1: torch.tensor([0, 1]), 2: torch.tensor([1, 0])},
        "concept_of": {0: 1, 1: 2},
        "drift_schedule": {0: 0, 1: 10},
    }
    evaluator._predict = lambda model: model
    predictions = torch.tensor([0, 1])
    before = evaluator.evaluate_staggered(predictions, None, 0)
    evaluator.data["drift_maps"][2] = torch.tensor([0, 1])
    assert evaluator.evaluate_staggered(predictions, None, 0) == before
    print("FEDDRIFT_SUPPORTED_PRIMITIVES_OK")


def check_feddaa_supported_primitives():
    cfg = Config()
    cfg.feddaa_tol_split = 0.3
    cfg.feddaa_tol_merge = 0.05
    cfg.feddaa_min_cluster_size = 2
    features = {
        0: np.asarray([1.0, 0.0]),
        1: np.asarray([0.9, 0.1]),
        2: np.asarray([-1.0, 0.0]),
        3: np.asarray([-0.9, -0.1]),
    }
    assignment, score, event = _feddaa_recluster(features, {i: 0 for i in features}, cfg)
    assert len(set(assignment.values())) == 2, (assignment, score, event)
    assert "split" in event, (assignment, score, event)
    weights = _softmax_neg([2.0, 0.5], 1.0)
    assert weights[1] > weights[0], weights
    one = {ci: np.ones(4) for ci in range(4)}
    assignment, score, _ = _feddaa_auto_assignment(one, cfg, 0)
    assert len(set(assignment.values())) == 1 and score == 0.0
    tiny = {**{ci: np.zeros(4) for ci in range(4)}, 4: np.ones(4) * 10}
    assignment, _, _ = _feddaa_auto_assignment(tiny, cfg, 0)
    assert len(set(assignment.values())) == 1
    reports = [LabelHistogramReport(0, 3, (3, 1)),
               LabelHistogramReport(1, 3, (1, 3))]
    assert np.allclose(_aggregate_label_reports(reports, 2, 3), [0.5, 0.5])
    try:
        _aggregate_label_reports(reports + reports[:1], 2, 3)
        raise AssertionError("duplicate label report accepted")
    except ValueError:
        pass
    assert not _feddaa_rebuild_due(46, 41, 6)
    assert _feddaa_rebuild_due(47, 41, 6)
    print("FEDDAA_SUPPORTED_PRIMITIVES_OK")


def check_simulated_secure_label_shift():
    # A roster change alone is not label drift: only client 0 participates,
    # and its current distribution is compared with its own baseline.
    roster_only = [LabelShiftReport(0, 3, (10, 0), (10, 0))]
    assert _simulated_secure_label_shift(roster_only, 2, 3) == 0.0

    changed = [LabelShiftReport(0, 3, (10, 0), (5, 5))]
    assert _simulated_secure_label_shift(changed, 2, 3) == 0.5
    try:
        _simulated_secure_label_shift(roster_only + roster_only, 2, 3)
        raise AssertionError("duplicate label shift report accepted")
    except ValueError:
        pass
    print("SIMULATED_SECURE_LABEL_SHIFT_OK")


def check_simulated_secure_input_shift():
    # Roster changes and unequal client sizes alone are not input drift.
    stable = [
        InputShiftReport(0, 3, 0.0, 100, 0.0, 100),
        InputShiftReport(1, 3, 900.0, 900, 900.0, 900),
    ]
    assert _simulated_secure_input_shift(stable, 3) == 0.0
    assert _simulated_secure_input_shift(stable[:1], 3) == 0.0
    shifted = [InputShiftReport(0, 3, 0.0, 100, 200.0, 100)]
    assert _simulated_secure_input_shift(shifted, 3) == 2.0
    try:
        _simulated_secure_input_shift(stable + stable[:1], 3)
        raise AssertionError("duplicate input shift report accepted")
    except ValueError:
        pass
    print("SIMULATED_SECURE_INPUT_SHIFT_OK")


def check_multiseed_acceptance_report():
    from multi_seed import build_acceptance_rows, case_name, target_agent_for

    assert case_name("drift", "label", "gradual") == "drift_gradual_label"
    assert case_name("dropout", dropout_mode="random", dropout_rate=.3) == "dropout_random_30"
    assert target_agent_for("drift_recurrent_real") is None

    def row(agent, worst, post, penalty):
        return {"case": "drift_label", "agent": agent,
                "primary_metric": "post_event_worst_acc",
                "post_event_worst_acc_mean": worst,
                "post_drift_mean_acc_mean": post,
                "final_acc_mean": post,
                "feddrift_cluster_purity_mean": "n/a",
                "misdiagnosis_penalty_mean": penalty}

    rows = [row("noop", .30, .60, .5), row("random", .27, .58, 1.0),
            row("label_prior_adapt", .32, .595, 0.0), row("oracle", .33, .61, 0.0),
            row("reweight", .305, .60, 1.0)]
    report = build_acceptance_rows(rows)[0]
    assert report["status"] == "conditional", report
    assert report["best_bad_agent"] == "reweight", report
    assert report["gap"] == .015, report

    rows[3]["post_event_worst_acc_mean"] = .315
    report = build_acceptance_rows(rows)[0]
    assert report["status"] == "conditional" and report["gap"] == .015, report

    def staggered(agent, post, final, penalty, purity="n/a"):
        return {"case": "staggered", "agent": agent, "primary_metric": "post_drift_mean_acc",
                "post_drift_mean_acc_mean": post, "final_acc_mean": final,
                "feddrift_cluster_purity_mean": purity, "misdiagnosis_penalty_mean": penalty}
    rows = [staggered("noop", .24, .32, .5), staggered("random", .30, .40, 1.0),
            staggered("spawn_concept_auto", .34, .48, 0.0, .98),
            staggered("spawn_concept", .37, .51, 1.0), staggered("oracle", .37, .51, 1.0)]
    report = build_acceptance_rows(rows)[0]
    assert report["status"] == "strong" and report["best_bad_agent"] == "random", report
    print("MULTISEED_ACCEPTANCE_REPORT_OK")


def check_sequence_scoring():
    from scoring import action_sequence_score

    hist = {"telemetry": [
        {"decision_due": True, "action": "no_op"},
        {"decision_due": True, "action": "drift_adapt"},
        {"decision_due": False, "action": "no_op"},
        {"decision_due": True, "action": "robust"},
    ]}
    actions, penalty = action_sequence_score(hist, ("drift_adapt",))
    assert actions == ["drift_adapt", "robust"] and penalty == 1.0, (actions, penalty)


for sc in ["drift", "fault", "dropout"]:
    run(sc)
check_dropout_window()
check_fdms_friend_discovery()
check_fdms_clustered_schedule()
check_drift_subtype_actions()
check_moment_align_metrics()
check_label_prior_metrics()
check_select_clients()
check_llm_error_fallback()
check_gradual_eval_mixture()
check_oort_supported_selector()
check_distance_trimmed_mean()
check_relative_fault_trigger()
check_feddrift_supported_primitives()
check_feddaa_supported_primitives()
check_simulated_secure_label_shift()
check_simulated_secure_input_shift()
check_multiseed_acceptance_report()
check_sequence_scoring()
print("LOGIC_OK")
