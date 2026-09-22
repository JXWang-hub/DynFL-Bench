"""Pure Phase 4A decision scoring; no Engine, Agent, or training imports."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
from pathlib import Path
from action_contract import ActionBundle, ActionState, reduce_action_state

from b3_audit import audit_calibration
from b3_fingerprint import git_identity, source_sha256
from b3_manifest import (
    DEFAULT_MANIFEST, load_manifest, manifest_sha256, validate_manifest,
    require_formal_manifest, require_versioned_output, response_window,
)
from config import PARTITION_MODES


CONFLICT_PENALTY = 0.25


def feddrift_structure_summary(telemetry, event_round: int) -> dict:
    """Reduce per-round FedDrift telemetry to stable experiment fields."""
    rows = [row for row in telemetry if int(row.get("round", -1)) >= event_round]

    def last_number(field):
        for row in reversed(rows):
            value = row.get(field, "n/a")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return round(float(value), 8)
        return "n/a"

    trigger = next((row.get("feddrift_first_trigger_round") for row in rows
                    if isinstance(row.get("feddrift_first_trigger_round"), (int, float))), None)
    action_round = next((int(row["round"]) for row in rows
                         if "spawn_concept_auto" in str(row.get("action_bundle", "")).split("|")), None)
    response_round = action_round if action_round is not None else trigger
    return {
        "feddrift_ari": last_number("feddrift_ari"),
        "feddrift_nmi": last_number("feddrift_nmi"),
        "feddrift_learner_count_error": last_number("feddrift_learner_count_error"),
        "feddrift_trigger_delay": (max(0, int(response_round) - event_round)
                                   if response_round is not None else "n/a"),
        "feddrift_assignment_churn": sum(
            int(row.get("feddrift_assignment_changed_clients", 0) or 0) for row in rows
        ),
        "feddrift_split_count": max(
            (int(row.get("feddrift_split_count", 0) or 0) for row in rows), default=0
        ),
        "feddrift_merge_count": sum(
            int(row.get("feddrift_merge_count", 0) or 0) for row in rows
        ),
    }


def _actions(value) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.split("|") if value else ()
    try:
        actions = tuple(value)
    except TypeError as exc:
        raise TypeError("bundle must be an iterable of action names") from exc
    if any(not isinstance(action, str) or not action for action in actions):
        raise ValueError("bundle actions must be non-empty strings")
    return actions or ("no_op",)


def _extra_penalty(action: str, causes: tuple[str, ...], matrix: dict,
                   default: float) -> float:
    rows = []
    for cause in causes:
        try:
            cause_row = matrix["causes"][cause]
            if not cause_row["informative"]:
                penalty = default
            else:
                penalty = float(cause_row["actions"][action]["penalty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"damage matrix is missing {cause}/{action}") from exc
        if not math.isfinite(penalty) or not 0.0 <= penalty <= 1.0:
            raise ValueError(f"invalid damage penalty for {cause}/{action}: {penalty}")
        rows.append(penalty)
    return max(rows)


def load_frozen_damage_matrix(matrix_path: Path, candidate_code_sha: str | None = None,
                              manifest_path: Path = DEFAULT_MANIFEST,
                              partition_mode: str = "noniid") -> tuple[dict, dict]:
    """Load only a complete Gate-P1B artifact chain bound to one candidate SHA."""
    matrix_path = Path(matrix_path)
    manifest_path = Path(manifest_path)
    trusted_git = git_identity(require_clean=True)
    if candidate_code_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", candidate_code_sha):
        raise ValueError("candidate_code_sha must be a lowercase 40-character Git SHA")
    if candidate_code_sha is not None and candidate_code_sha != trusted_git["git_sha"]:
        raise ValueError("candidate_code_sha does not match the trusted checkout")
    plan_path = matrix_path.with_name("atomic_action_calibration_plan.json")
    for path in (manifest_path, plan_path, matrix_path):
        if not path.is_file():
            raise ValueError(f"required frozen artifact is missing: {path}")
    try:
        manifest = load_manifest(manifest_path)
        require_versioned_output(matrix_path, manifest, partition_mode)
        require_formal_manifest(manifest)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        raw_name = matrix["source_raw_file"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen calibration artifact") from exc
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("damage matrix has an invalid raw filename")
    raw_path = matrix_path.with_name(raw_name)
    if not raw_path.is_file():
        raise ValueError(f"required frozen artifact is missing: {raw_path}")
    audit_calibration(
        manifest, plan, raw_path, matrix, root=manifest_path.resolve().parent,
        expected_code_sha=trusted_git["git_sha"],
    )
    if plan.get("partition_mode") != partition_mode or matrix.get("partition_mode") != partition_mode:
        raise ValueError("damage matrix partition mode mismatch")
    if matrix.get("source_sha256") != source_sha256():
        raise ValueError("damage matrix source fingerprint mismatch")
    return manifest, matrix


def effective_bundles_from_telemetry(telemetry, manifest: dict,
                                     event_round: int) -> list[list[str]]:
    """Replay the shared reducer and return each decision's declared next state."""
    start, end = response_window(manifest, event_round)
    by_round = _effective_bundles_by_round(telemetry, end)
    return [by_round[round_idx] for round_idx in range(start, end + 1)]


def _effective_bundles_by_round(telemetry, last_round: int) -> dict[int, list[str]]:
    """Replay action state once so overlapping windows share identical rounds."""
    rows = {int(row["round"]): row for row in telemetry}
    if len(rows) != len(telemetry) or any(round_idx not in rows for round_idx in range(last_round + 1)):
        raise ValueError("telemetry does not cover the response window exactly once")
    state, result = ActionState(), {}
    for round_idx in range(last_round + 1):
        row = rows[round_idx]
        if row.get("decision_due", True):
            names = _actions(row.get("activated_bundle", row.get("action_bundle", "no_op")))
            try:
                emitted = ActionBundle(names)
            except (TypeError, ValueError) as exc:
                raise ValueError("telemetry contains an invalid activated bundle") from exc
        else:
            emitted = None
        transition = reduce_action_state(state, emitted)
        for key, expected in (("active_before", transition.active_before),
                              ("active_after", transition.active_after)):
            if key in row and _actions(row[key]) != (expected or ("no_op",)):
                raise ValueError(f"telemetry {key} does not match shared action reducer")
        state = transition.state
        result[round_idx] = list(transition.active_after) or ["no_op"]
    return result


def _causes_at_round(episode_plan: dict, round_idx: int) -> tuple[str, ...]:
    try:
        hidden = episode_plan["hidden_event"]
        causes = hidden["active_causes"]
        intervals = hidden["cause_intervals"]
    except (KeyError, TypeError) as exc:
        raise ValueError("episode plan is missing scorer-side cause intervals") from exc
    active = []
    for cause in causes:
        try:
            spans = intervals[cause].values()
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"episode plan is missing intervals for {cause}") from exc
        if any(start <= round_idx and (end is None or round_idx < end)
               for client_spans in spans for start, end in client_spans):
            active.append(cause)
    return tuple(active)


def _active_causes_at_round(episode_plan: dict, round_idx: int) -> tuple[str, ...]:
    active = _causes_at_round(episode_plan, round_idx)
    if not active:
        raise ValueError(f"response round {round_idx} has no active cause")
    return active


def _stable_metrics(episode_plan: dict, bundles: dict[int, list[str]]) -> dict:
    first_event = min(int(row["round"])
                      for row in episode_plan["hidden_event"]["event_schedule"])
    stable = [round_idx for round_idx in sorted(bundles)
              if round_idx >= first_event and not _causes_at_round(episode_plan, round_idx)]
    if not stable:
        return {"stable_rounds": 0, "stable_over_intervention": "n/a",
                "stable_noop_correctness": "n/a", "stable_action_churn": "n/a",
                "action_close_delay": "n/a"}
    active = {round_idx: bundles[round_idx] != ["no_op"] for round_idx in stable}
    segments = []
    for round_idx in stable:
        if not segments or round_idx != segments[-1][-1] + 1:
            segments.append([])
        segments[-1].append(round_idx)
    delays = []
    for segment in segments:
        delays.append(next((round_idx - segment[0] for round_idx in segment
                            if not active[round_idx]), len(segment)))
    return {
        "stable_rounds": len(stable),
        "stable_over_intervention": sum(active.values()),
        "stable_noop_correctness": round(1.0 - sum(active.values()) / len(stable), 8),
        "stable_action_churn": sum(
            bundles[current] != bundles[previous]
            for segment in segments
            for previous, current in zip(segment, segment[1:])
        ),
        "action_close_delay": round(statistics.fmean(delays), 8),
    }


def score_history(episode_plan: dict, history: dict, manifest: dict, damage_matrix: dict,
                  u_oracle: float, u_noop: float) -> dict:
    """Bind scorer-only truth to Engine telemetry without exposing it publicly."""
    try:
        hidden = history["hidden_event_log"]
        event = hidden[0]
        event_round = min(int(item["round"]) for item in episode_plan["hidden_event"]["event_schedule"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("history is missing the scorer-side hidden event log") from exc
    expected = episode_plan["hidden_event"]
    if len(hidden) != 1 or any((
        event.get("episode_id") != episode_plan.get("episode_id"),
        event.get("plan_sha256") != episode_plan.get("plan_sha256"),
        event.get("partition_sha256") != episode_plan.get("partition_sha256"),
        event.get("active_causes") != expected.get("active_causes"),
        event.get("affected_clients") != expected.get("affected_clients"),
        event.get("cause_intervals") != expected.get("cause_intervals"),
        event.get("event_schedule") != expected.get("event_schedule"),
        event.get("event_round") != event_round,
    )):
        raise ValueError("history hidden event does not match the episode plan")
    schedule = expected["event_schedule"]
    telemetry = history.get("telemetry", ())
    bundles = _effective_bundles_by_round(telemetry, len(telemetry) - 1)
    score = score_scheduled_episode(episode_plan, bundles, manifest, damage_matrix)
    window_rounds = set()
    for item in schedule:
        start, end = response_window(manifest, int(item["round"]))
        window_rounds.update(range(start, end + 1))
    window_rounds = sorted(window_rounds)
    score["action_churn"] = sum(
        bundles[current] != bundles[previous]
        for previous, current in zip(window_rounds, window_rounds[1:])
        if current == previous + 1
    )
    effect_bundles = {
        int(row["round"]): list(_actions(row.get("active_before", bundles[int(row["round"])])))
        for row in telemetry
    }
    score.update(_stable_metrics(episode_plan, effect_bundles))
    try:
        utility = statistics.fmean(float(value) for value in history["acc"][event_round:])
    except (KeyError, TypeError, ValueError, statistics.StatisticsError) as exc:
        raise ValueError("history has no valid post-event utility") from exc
    ratio = regret_ratio(u_oracle, utility, u_noop, manifest["calibration"]["epsilon"])
    score.update({
        "post_event_mean_accuracy": round(utility, 8),
        "regret_ratio": ratio if ratio == "n/a" else round(float(ratio), 8),
        "result_check": "uninformative" if ratio == "n/a" else "informative",
    })
    return score


def public_episode_score(score: dict) -> dict:
    """Allowlist public leaderboard fields; cause/action truth stays scorer-side."""
    fields = (
        "episode_id", "response_rounds", "decision_score", "cause_recall",
        "action_precision", "bundle_f1", "time_to_first_valid_action",
        "time_to_full_cause_coverage", "over_intervention", "conflict_count",
        "action_churn", "stable_rounds", "stable_over_intervention",
        "stable_noop_correctness", "stable_action_churn",
        "action_close_delay", "post_event_mean_accuracy", "regret_ratio", "result_check",
    )
    return {field: score[field] for field in fields}


def score_bundle(active_causes, bundle, manifest: dict, damage_matrix: dict) -> dict:
    """Score one effective bundle against the active hidden causes."""
    causes = tuple(active_causes)
    if not causes or len(causes) != len(set(causes)):
        raise ValueError("active_causes must contain distinct cause names")
    unknown_causes = set(causes) - set(manifest.get("causes", {}))
    if unknown_causes:
        raise ValueError(f"unknown active causes: {sorted(unknown_causes)}")
    expected_sha = manifest_sha256(manifest)
    if damage_matrix.get("manifest_sha256") != expected_sha:
        raise ValueError("damage matrix manifest SHA mismatch")

    raw_actions = _actions(bundle)
    unknown_actions = set(raw_actions) - set(manifest.get("actions", {}))
    if unknown_actions:
        raise ValueError(f"unknown actions: {sorted(unknown_actions)}")
    actions = tuple(dict.fromkeys(raw_actions))
    feasible = {
        cause: set(manifest["causes"][cause]["feasible_actions"])
        for cause in causes
    }
    all_feasible = set().union(*feasible.values())
    covered = tuple(cause for cause in causes if feasible[cause].intersection(actions))
    missed = tuple(cause for cause in causes if cause not in covered)
    active_actions = tuple(action for action in actions if action != "no_op")
    extras = tuple(action for action in active_actions if action not in all_feasible)

    duplicates = len(raw_actions) - len(actions)
    no_op_conflict = int("no_op" in actions and len(actions) > 1)
    families = [manifest["actions"][action]["family"] for action in active_actions]
    family_conflicts = len(families) - len(set(families))
    conflict_count = duplicates + no_op_conflict + family_conflicts

    calibration = manifest["calibration"]
    missed_total = len(missed) * float(calibration["missed_cause_penalty"])
    extra_total = sum(
        _extra_penalty(
            action, causes, damage_matrix,
            float(calibration["uninformative_penalty"]),
        )
        for action in extras
    )
    conflict_total = conflict_count * CONFLICT_PENALTY
    penalty = min(1.0, (missed_total + extra_total + conflict_total) / len(causes))

    precision = ((len(active_actions) - len(extras)) / len(active_actions)
                 if active_actions else 0.0)
    recall = len(covered) / len(causes)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "chosen_bundle": list(raw_actions),
        "covered_causes": list(covered),
        "missed_causes": list(missed),
        "extra_actions": list(extras),
        "conflict_count": conflict_count,
        "missed_penalty": round(missed_total, 8),
        "extra_penalty": round(extra_total, 8),
        "conflict_penalty": round(conflict_total, 8),
        "bundle_penalty": round(penalty, 8),
        "cause_recall": round(recall, 8),
        "action_precision": round(precision, 8),
        "bundle_f1": round(f1, 8),
    }


def score_episode(episode_plan: dict, effective_bundles, manifest: dict,
                  damage_matrix: dict) -> dict:
    """Score effective bundles in response-window order (delay is zero-based)."""
    expected_sha = manifest_sha256(manifest)
    if episode_plan.get("suite_manifest_sha256") != expected_sha:
        raise ValueError("episode plan manifest SHA mismatch")
    if isinstance(effective_bundles, str):
        raise TypeError("effective_bundles must be a response-window sequence")
    bundles = tuple(effective_bundles)
    if not bundles:
        raise ValueError("response window must contain at least one bundle")
    try:
        causes = episode_plan["hidden_event"]["active_causes"]
    except (KeyError, TypeError) as exc:
        raise ValueError("episode plan is missing hidden active causes") from exc
    rows = [score_bundle(causes, bundle, manifest, damage_matrix) for bundle in bundles]
    return _summarize_rows(episode_plan.get("episode_id"), rows)


def _summarize_rows(episode_id, rows) -> dict:
    count = len(rows)
    mean = lambda key: sum(row[key] for row in rows) / count
    first_valid = next((i for i, row in enumerate(rows) if row["covered_causes"]), None)
    full_coverage = next((i for i, row in enumerate(rows) if not row["missed_causes"]), None)
    episode_penalty = mean("bundle_penalty")
    return {
        "episode_id": episode_id,
        "response_rounds": count,
        "episode_penalty": round(episode_penalty, 8),
        "decision_score": round(100.0 * (1.0 - episode_penalty), 8),
        "cause_recall": round(mean("cause_recall"), 8),
        "action_precision": round(mean("action_precision"), 8),
        "bundle_f1": round(mean("bundle_f1"), 8),
        "time_to_first_valid_action": first_valid,
        "time_to_full_cause_coverage": full_coverage,
        "over_intervention": sum(len(row["extra_actions"]) for row in rows),
        "conflict_count": sum(row["conflict_count"] for row in rows),
        "per_round": rows,
    }


def score_scheduled_episode(episode_plan: dict, effective_bundles_by_round: dict,
                            manifest: dict, damage_matrix: dict) -> dict:
    """Score each event window independently, then macro-average event scores."""
    if episode_plan.get("suite_manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("episode plan manifest SHA mismatch")
    try:
        schedule = episode_plan["hidden_event"]["event_schedule"]
    except (KeyError, TypeError) as exc:
        raise ValueError("episode plan is missing its event schedule") from exc
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("event schedule must contain at least one event")

    event_scores = []
    for item in sorted(schedule, key=lambda value: int(value["round"])):
        event_round = int(item["round"])
        start, end = response_window(manifest, event_round)
        try:
            rows = [score_bundle(
                _active_causes_at_round(episode_plan, round_idx),
                effective_bundles_by_round[round_idx], manifest, damage_matrix,
            ) for round_idx in range(start, end + 1)]
        except KeyError as exc:
            raise ValueError("effective bundles do not cover every event window") from exc
        score = _summarize_rows(episode_plan.get("episode_id"), rows)
        score.update({"event_round": event_round, "response_window": [start, end]})
        event_scores.append(score)

    mean = lambda key: statistics.fmean(score[key] for score in event_scores)
    delay = lambda key: (None if any(score[key] is None for score in event_scores)
                         else statistics.fmean(score[key] for score in event_scores))
    return {
        "episode_id": episode_plan.get("episode_id"),
        "event_count": len(event_scores),
        "response_rounds": sum(score["response_rounds"] for score in event_scores),
        "episode_penalty": round(mean("episode_penalty"), 8),
        "decision_score": round(mean("decision_score"), 8),
        "cause_recall": round(mean("cause_recall"), 8),
        "action_precision": round(mean("action_precision"), 8),
        "bundle_f1": round(mean("bundle_f1"), 8),
        "time_to_first_valid_action": delay("time_to_first_valid_action"),
        "time_to_full_cause_coverage": delay("time_to_full_cause_coverage"),
        "over_intervention": sum(score["over_intervention"] for score in event_scores),
        "conflict_count": sum(score["conflict_count"] for score in event_scores),
        "per_round": [row for score in event_scores for row in score["per_round"]],
        "per_event": event_scores,
    }


def regret_ratio(u_oracle: float, u_agent: float, u_noop: float,
                 epsilon: float = 0.01) -> float | str:
    u_oracle, u_agent, u_noop, epsilon = map(
        float, (u_oracle, u_agent, u_noop, epsilon)
    )
    if not all(map(math.isfinite, (u_oracle, u_agent, u_noop, epsilon))) or epsilon <= 0:
        raise ValueError("utilities must be finite and epsilon must be positive")
    denominator = u_oracle - u_noop
    return "n/a" if denominator < epsilon else (u_oracle - u_agent) / denominator


def _fixture_matrix(manifest: dict, partition_mode: str = "noniid") -> dict:
    causes = {}
    for cause in manifest["calibration"]["causes"]:
        feasible = set(manifest["causes"][cause]["feasible_actions"])
        actions = {}
        for action in manifest["actions"]:
            penalty = 0.0 if action in feasible else 0.5 if action == "no_op" else 0.25
            actions[action] = {"penalty": penalty}
        causes[cause] = {"informative": True, "actions": actions}
    causes["real_drift"]["actions"]["robust"]["penalty"] = 0.8
    causes["dropout"]["actions"]["robust"]["penalty"] = 0.6
    return {"manifest_sha256": manifest_sha256(manifest),
            "partition_mode": partition_mode,
            "matrix_sha256": "fixture", "causes": causes}


def self_check(manifest_path: Path = DEFAULT_MANIFEST) -> None:
    if not __debug__:
        raise RuntimeError("scoring self-check refuses optimized mode")
    from b3_episode_plan import build_episode_plan

    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    assert not errors, "; ".join(errors)
    plan = build_episode_plan("real_dropout", 0, manifest_path=manifest_path)
    matrix = _fixture_matrix(manifest)
    causes = plan["hidden_event"]["active_causes"]
    canonical = plan["hidden_event"]["canonical_bundle"]

    oracle = score_episode(plan, [canonical, canonical], manifest, matrix)
    assert oracle["episode_penalty"] == 0.0 and oracle["decision_score"] == 100.0
    alternative = score_bundle(
        causes, ["drift_adapt", "friend_substitute"], manifest, matrix
    )
    assert alternative["missed_causes"] == [] and alternative["bundle_penalty"] == 0.0

    missed = score_bundle(causes, ["drift_adapt"], manifest, matrix)
    assert missed["missed_causes"] == ["dropout"] and missed["bundle_penalty"] == 0.25

    extra = score_bundle(
        causes, ["drift_adapt", "dropout_handle", "robust"], manifest, matrix
    )
    assert extra["extra_actions"] == ["robust"]
    assert extra["extra_penalty"] == 0.8 and extra["bundle_penalty"] == 0.4

    uninformative = copy.deepcopy(matrix)
    uninformative["causes"]["dropout"]["informative"] = False
    assert score_bundle(
        causes, ["drift_adapt", "dropout_handle", "robust"], manifest, uninformative
    )["extra_penalty"] == 0.8

    delayed = score_episode(
        plan, [["no_op"], ["drift_adapt"], canonical], manifest, matrix
    )
    assert delayed["decision_score"] == 75.0
    assert delayed["time_to_first_valid_action"] == 1
    assert delayed["time_to_full_cause_coverage"] == 2
    assert math.isclose(regret_ratio(0.8, 0.7, 0.6), 0.5)
    assert regret_ratio(0.505, 0.502, 0.5) == "n/a"
    assert response_window(manifest, 50) == (50, 59)
    structure = feddrift_structure_summary([
        {"round": 40, "action_bundle": "spawn_concept_auto",
         "feddrift_ari": "n/a", "feddrift_assignment_changed_clients": 2,
         "feddrift_split_count": 1, "feddrift_merge_count": 0},
        {"round": 41, "action_bundle": "no_op", "feddrift_ari": 0.8,
         "feddrift_nmi": 0.9, "feddrift_learner_count_error": 0,
         "feddrift_assignment_changed_clients": 1,
         "feddrift_split_count": 1, "feddrift_merge_count": 1},
    ], 40)
    assert structure == {
        "feddrift_ari": 0.8, "feddrift_nmi": 0.9,
        "feddrift_learner_count_error": 0.0, "feddrift_trigger_delay": 0,
        "feddrift_assignment_churn": 3, "feddrift_split_count": 1,
        "feddrift_merge_count": 1,
    }

    scheduled = copy.deepcopy(plan)
    scheduled["hidden_event"]["event_schedule"] = [
        {"round": 43, "activate": causes, "severity": {}},
        {"round": 40, "activate": causes, "severity": {}},
    ]
    scheduled["hidden_event"]["cause_intervals"] = {
        cause: {"0": [[40, None]]} for cause in causes
    }
    by_round = {round_idx: canonical for round_idx in range(40, 53)}
    overlap = score_scheduled_episode(scheduled, by_round, manifest, matrix)
    assert overlap["event_count"] == 2 and overlap["response_rounds"] == 20
    assert overlap["decision_score"] == 100.0
    assert [row["response_window"] for row in overlap["per_event"]] == [[40, 49], [43, 52]]
    history = {
        "hidden_event_log": [{
            "episode_id": scheduled["episode_id"],
            "plan_sha256": scheduled["plan_sha256"],
            "partition_sha256": scheduled.get("partition_sha256"),
            "event_round": 40,
            "active_causes": scheduled["hidden_event"]["active_causes"],
            "affected_clients": scheduled["hidden_event"]["affected_clients"],
            "cause_intervals": scheduled["hidden_event"]["cause_intervals"],
            "event_schedule": scheduled["hidden_event"]["event_schedule"],
        }],
        "telemetry": [{
            "round": round_idx,
            "action_bundle": "|".join(canonical) if round_idx >= 40 else "no_op",
        } for round_idx in range(53)],
        "acc": [0.8] * 53,
    }
    replayed = score_history(scheduled, history, manifest, matrix, 0.8, 0.5)
    assert replayed["decision_score"] == 100.0 and len(replayed["per_round"]) == 20

    stable_plan = copy.deepcopy(plan)
    stable_plan["hidden_event"]["event_schedule"] = [
        {"round": 40, "activate": causes, "severity": {}}
    ]
    stable_plan["hidden_event"]["cause_intervals"] = {
        cause: {"0": [[40, 50]]} for cause in causes
    }
    stable_hidden = stable_plan["hidden_event"]
    stable_history = {
        "hidden_event_log": [{
            "episode_id": stable_plan["episode_id"],
            "plan_sha256": stable_plan["plan_sha256"],
            "partition_sha256": stable_plan.get("partition_sha256"),
            "event_round": 40,
            "active_causes": stable_hidden["active_causes"],
            "affected_clients": stable_hidden["affected_clients"],
            "cause_intervals": stable_hidden["cause_intervals"],
            "event_schedule": stable_hidden["event_schedule"],
        }],
        "telemetry": [{
            "round": round_idx,
            "action_bundle": ("|".join(canonical)
                              if 40 <= round_idx < 51 else "no_op"),
        } for round_idx in range(53)],
        "acc": [0.8] * 53,
    }
    stable_score = score_history(stable_plan, stable_history, manifest, matrix, 0.8, 0.5)
    assert stable_score["stable_rounds"] == 3
    assert stable_score["stable_over_intervention"] == 1
    assert stable_score["stable_noop_correctness"] == 0.66666667
    assert stable_score["stable_action_churn"] == 1
    assert stable_score["action_close_delay"] == 1.0

    conflict = score_bundle(
        causes, ["drift_adapt", "drift_adapt"], manifest, matrix
    )
    assert conflict["conflict_count"] == 1
    telemetry = [
        {"round": round_idx, "action_bundle": (
            "drift_adapt|dropout_handle" if round_idx >= 40 else "no_op"
        )}
        for round_idx in range(50)
    ]
    effective = effective_bundles_from_telemetry(telemetry, manifest, 40)
    assert effective == [["dropout_handle", "drift_adapt"]] * 10
    public = public_episode_score({
        **oracle,
        "action_churn": 0,
        "stable_rounds": 0,
        "stable_over_intervention": "n/a",
        "stable_noop_correctness": "n/a",
        "stable_action_churn": "n/a",
        "action_close_delay": "n/a",
        "post_event_mean_accuracy": 0.8,
        "regret_ratio": 0.0,
        "result_check": "informative",
    })
    assert not ({"per_round", "missed_causes", "covered_causes", "active_causes"} & set(public))
    print(
        "B3_PHASE4A_OK"
        f" oracle_score={oracle['decision_score']:.0f}"
        f" missed_penalty={missed['bundle_penalty']:.2f}"
        f" extra_penalty={extra['extra_penalty']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--candidate-code-sha")
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="noniid")
    args = parser.parse_args()
    self_check(args.manifest)
    if args.matrix:
        _, matrix = load_frozen_damage_matrix(
            args.matrix, args.candidate_code_sha, args.manifest, args.partition_mode
        )
        print(
            "B3_PHASE4B_MATRIX_OK"
            f" jobs={matrix['run_count']}"
            f" matrix_sha256={matrix['matrix_sha256']}"
            f" code_sha={args.candidate_code_sha}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
