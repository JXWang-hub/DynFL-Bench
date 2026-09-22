"""State-transition Agent decision audit for DynFL-Bench; stdlib-only."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict

from action_contract import ACTION_SPECS, ActionBundle


CORRECT_LABELS = {"CORRECT", "NO_OP_CORRECT"}
COVERAGE_RANK = {"NONE": 0, "PARTIAL": 1, "FULL": 2}
BUNDLE_STATUSES = {"VALID", "INVALID_SCHEMA", "UNKNOWN_ACTION", "CONFLICT"}

EVENT_FIELDS = [
    "run_id", "agent_id", "agent_config_id", "episode_id", "event_id",
    "case_id", "seed", "partition_mode", "transition_type", "event_round",
    "lifecycle_transition_type",
    "first_due_round", "next_transition_round", "active_causes_before",
    "active_causes_after", "removed_causes", "affected_clients", "severity",
    "canonical_bundle", "emit_deadline", "correct_decision_round",
    "correct_decision_bundle", "decision_delay", "final_bundle_before_transition",
    "final_decision_label", "decision_count", "correct_decision_count",
    "invalid_bundle_count", "over_intervention_decision_count",
    "bundle_change_count", "post_correct_regression_count", "best_coverage",
    "final_coverage", "event_outcome_bucket", "ever_correct", "ended_correct",
    "never_correct", "timing_status", "missing_causes_final",
    "unnecessary_actions_seen", "ever_over_intervened",
    "final_has_unnecessary_actions", "audit_conclusion",
]

DECISION_FIELDS = [
    "run_id", "agent_id", "agent_config_id", "episode_id", "event_id",
    "decision_round", "active_causes", "public_observation_ref", "raw_output_ref",
    "parsed_bundle", "bundle_status", "covered_causes", "missing_causes",
    "unnecessary_actions", "coverage_status", "decision_label", "relative_delay",
    "deadline_relation",
]

SUMMARY_FIELDS = [
    "agent_id", "agent_config_id", "agent_version", "model", "policy",
    "evaluated_runs", "evaluated_events", "ever_correct_event_count",
    "ever_correct_event_rate", "ended_correct_event_rate",
    "never_correct_event_count", "full_with_extra_best_event_count",
    "partial_best_event_count", "none_best_event_count", "unassessed_event_count",
    "correct_decision_rate", "invalid_bundle_rate",
    "ever_over_intervened_event_rate", "over_intervention_decision_rate",
    "timed_event_count", "on_time_event_rate", "median_decision_delay",
    "worst_decision_delay", "untimed_event_count", "bundle_change_count",
    "post_correct_regression_count", "cause_removal_event_count",
    "ever_correct_deescalation_rate", "ended_correct_deescalation_rate",
    "on_time_deescalation_rate", "never_deescalated_count",
    "median_deescalation_delay", "init_decision_count", "init_no_op_correct_rate",
    "init_over_intervention_rate",
]

INTEGRITY_FIELDS = [
    "run_id", "agent_id", "agent_config_id", "episode_id", "run_status",
    "failure_type", "failure_round", "rerun_required",
]


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value) -> str:
    text = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_run_id(metadata: dict) -> str:
    """Create a stable report id from the frozen run inputs."""
    return "run_" + _sha256(metadata)[:32]


def agent_config_id(agent_spec: dict) -> str:
    return "agentcfg_" + _sha256(agent_spec)[:24]


def _pipe(values) -> str:
    return "|".join(str(value) for value in values)


def _actions(value) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.split("|") if part)
    return tuple(value)


def _active_causes(plan: dict, round_idx: int) -> tuple[str, ...]:
    active = []
    intervals = plan["hidden_event"].get("cause_intervals", {})
    for cause, by_client in intervals.items():
        spans = (span for client_spans in by_client.values() for span in client_spans)
        if any(int(start) <= round_idx and (end is None or round_idx < int(end))
               for start, end in spans):
            active.append(cause)
    return tuple(active)


def _transition_type(before: tuple[str, ...] | None,
                     after: tuple[str, ...]) -> str:
    if before is None:
        return "INIT"
    before_set, after_set = set(before), set(after)
    if before_set < after_set:
        return "ACTIVATE"
    if before_set and not after_set:
        return "STABLE"
    return "CHANGE"


def _lifecycle_transition_type(before: tuple[str, ...] | None,
                               after: tuple[str, ...]) -> str:
    if before is None:
        return "INIT"
    before_set, after_set = set(before), set(after)
    if not after_set:
        return "CLEAR"
    if before_set < after_set:
        return "ADD"
    if after_set < before_set:
        return "REMOVE"
    return "CHANGE"


def _event_metadata(plan: dict, manifest: dict, round_idx: int,
                    active: tuple[str, ...]) -> tuple[dict, dict, int | None]:
    schedule = plan["hidden_event"].get("event_schedule", [])
    severity = {}
    scheduled = None
    for item in sorted(schedule, key=lambda row: int(row["round"])):
        if int(item["round"]) > round_idx:
            break
        for cause, value in item.get("severity", {}).items():
            severity[cause] = value
        if int(item["round"]) == round_idx:
            scheduled = item
    affected = {
        cause: plan["hidden_event"].get("affected_clients", {}).get(cause)
        for cause in active
    }
    deadline = None
    if scheduled is not None:
        deadline = scheduled.get("emit_deadline")
        if deadline is None and manifest.get("protocol", {}).get("response_window_length"):
            deadline = round_idx + int(manifest["protocol"]["response_window_length"]) - 1
        deadline = None if deadline is None else int(deadline)
    return ({cause: affected[cause] for cause in active},
            {cause: severity.get(cause) for cause in active}, deadline)


def compile_events(plan: dict, manifest: dict, telemetry: list[dict]) -> list[dict]:
    """Compile INIT and every active-cause set transition from cause intervals."""
    planned_rounds = int(plan["total_rounds"])
    if planned_rounds <= 0:
        raise ValueError("episode must contain at least one round")
    by_round = {int(row["round"]): row for row in telemetry}
    total_rounds = max(by_round, default=-1) + 1
    if (not total_rounds or total_rounds > planned_rounds or
            set(by_round) != set(range(total_rounds))):
        raise ValueError("telemetry must cover every episode round exactly once")

    states = [_active_causes(plan, round_idx) for round_idx in range(total_rounds)]
    starts = [0] + [round_idx for round_idx in range(1, total_rounds)
                    if states[round_idx] != states[round_idx - 1]]
    events = []
    scheduled_rounds = {int(row["round"])
                        for row in plan["hidden_event"].get("event_schedule", [])}
    if any(round_idx not in starts for round_idx in scheduled_rounds if round_idx != 0):
        raise ValueError("parameter-only scheduled changes are unsupported")

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else total_rounds
        before = None if index == 0 else states[start - 1]
        after = states[start]
        due = [round_idx for round_idx in range(start, end)
               if bool(by_round[round_idx].get("decision_due"))]
        if not due:
            raise ValueError(f"event at round {start} has no Agent call opportunity")
        affected, severity, deadline = _event_metadata(plan, manifest, start, after)
        if deadline is not None:
            if deadline >= end:
                raise ValueError(f"deadline at round {deadline} crosses the next transition")
            if not any(round_idx <= deadline for round_idx in due):
                raise ValueError(f"event at round {start} has no call before deadline")
        events.append({
            "event_id": f"E{index:02d}",
            "transition_type": _transition_type(before, after),
            "event_round": start,
            "first_due_round": due[0],
            "next_transition_round": None if end == total_rounds else end,
            "end_round": end,
            "active_causes_before": () if before is None else before,
            "active_causes_after": after,
            "removed_causes": () if before is None else tuple(
                cause for cause in before if cause not in set(after)
            ),
            "affected_clients": affected,
            "severity": severity,
            "emit_deadline": deadline,
        })
    return events


def _artifact(payload, artifacts: dict[str, object]) -> str:
    digest = _sha256(payload)
    ref = "sha256:" + digest
    artifacts.setdefault(ref, payload)
    return ref


def _canonical_bundle(active_causes: tuple[str, ...], manifest: dict,
                      completed: set[str]) -> tuple[str, ...]:
    actions = []
    for cause in active_causes:
        feasible = manifest["causes"][cause]["feasible_actions"]
        if any(action in completed for action in feasible
               if ACTION_SPECS[action].lifecycle == "one_shot_irreversible"):
            continue
        action = manifest["causes"][cause].get("canonical_action")
        if action and action not in actions:
            actions.append(action)
    return tuple(actions or ("no_op",))


def _score_decision(active_causes: tuple[str, ...], actions: tuple[str, ...],
                    bundle_status: str, manifest: dict,
                    completed: set[str]) -> dict:
    if bundle_status != "VALID":
        return {
            "covered": (), "missing": (), "unnecessary": (), "coverage": "",
            "label": "INVALID_BUNDLE", "comparison": None,
        }
    if not actions:
        raise ValueError("VALID decision has no parsed bundle")

    # Revalidate the persisted declaration rather than trusting telemetry alone.
    if len(actions) != len(set(actions)):
        raise ValueError("VALID decision contains duplicate actions")
    ActionBundle(actions)
    if any(action not in manifest["actions"] for action in actions):
        raise ValueError("VALID decision contains an action outside the suite manifest")

    satisfied = set()
    pending = []
    for cause in active_causes:
        feasible = set(manifest["causes"][cause]["feasible_actions"])
        if any(action in completed and
               ACTION_SPECS[action].lifecycle == "one_shot_irreversible"
               for action in feasible):
            satisfied.add(cause)
        else:
            pending.append(cause)

    selected = tuple(action for action in actions if action != "no_op")
    feasible_by_cause = {
        cause: set(manifest["causes"][cause]["feasible_actions"])
        for cause in pending
    }
    allowed = set().union(*feasible_by_cause.values()) if feasible_by_cause else set()
    covered = set(satisfied)
    for cause, feasible in feasible_by_cause.items():
        if feasible.intersection(selected):
            covered.add(cause)
    missing = tuple(cause for cause in active_causes if cause not in covered)
    unnecessary = tuple(action for action in selected
                        if action in completed or action not in allowed)
    covered_ordered = tuple(cause for cause in active_causes if cause in covered)
    coverage = ("FULL" if actions == ("no_op",) else "NONE") if not active_causes else (
        "FULL" if not missing else "PARTIAL" if covered_ordered else "NONE"
    )

    if not active_causes:
        label = "NO_OP_CORRECT" if actions == ("no_op",) else "OVER_INTERVENTION"
    elif coverage == "FULL" and not unnecessary:
        label = "CORRECT"
    elif coverage == "FULL":
        label = "OVER_INTERVENTION"
    elif coverage == "PARTIAL" and not unnecessary:
        label = "PARTIAL"
    elif coverage == "PARTIAL":
        label = "PARTIAL_WITH_EXTRA"
    elif actions == ("no_op",):
        label = "UNDER_INTERVENTION"
    else:
        label = "WRONG_ACTION"

    comparison = tuple(action for action in actions
                       if action not in completed or action == "no_op")
    if not comparison:
        comparison = ("no_op",)
    for cause in active_causes:
        feasible = set(manifest["causes"][cause]["feasible_actions"])
        for action in selected:
            if (action in feasible and
                    ACTION_SPECS[action].lifecycle == "one_shot_irreversible"):
                completed.add(action)
    return {
        "covered": covered_ordered, "missing": missing,
        "unnecessary": unnecessary, "coverage": coverage,
        "label": label, "comparison": comparison,
    }


def _deadline_relation(round_idx: int, deadline: int | None) -> str:
    if deadline is None:
        return "UNSET"
    if round_idx < deadline:
        return "BEFORE"
    if round_idx == deadline:
        return "ON"
    return "AFTER"


def _invalid_integrity(run_id: str, agent_id: str, config_id: str,
                       episode_id: str, failure_type: str,
                       failure_round="") -> dict:
    return {
        "run_id": run_id, "agent_id": agent_id, "agent_config_id": config_id,
        "episode_id": episode_id, "run_status": "INVALID",
        "failure_type": failure_type, "failure_round": failure_round,
        "rerun_required": True,
    }


def audit_episode(run_id: str, agent_id: str, agent_spec: dict, plan: dict,
                  history: dict, manifest: dict) -> dict:
    """Audit one Agent × episode. Invalid runs return no formal decision/event rows."""
    config_id = agent_config_id(agent_spec)
    episode_id = plan["episode_id"]
    telemetry = history.get("telemetry", [])
    required = {
        "round", "decision_due", "decision_bundle_status", "declared_bundle",
        "raw_output", "public_observation_json", "agent_interface_error",
    }
    for row in telemetry:
        if not row.get("decision_due"):
            continue
        missing = required - set(row)
        if missing:
            return {
                "integrity": _invalid_integrity(
                    run_id, agent_id, config_id, episode_id,
                    "AUDIT_FIELDS_MISSING", row.get("round", ""),
                ), "events": [], "decisions": [], "artifacts": {},
            }
        interface_error = (row.get("agent_interface_error") or
                           row.get("llm_api_error") or row.get("llm_budget_error"))
        if interface_error:
            failure = ("AGENT_BUDGET_EXHAUSTED" if row.get("llm_budget_error")
                       else "AGENT_INTERFACE_FAILED")
            return {
                "integrity": _invalid_integrity(
                    run_id, agent_id, config_id, episode_id, failure, int(row["round"]),
                ), "events": [], "decisions": [], "artifacts": {},
            }

    try:
        events = compile_events(plan, manifest, telemetry)
    except ValueError:
        raise
    by_round = {int(row["round"]): row for row in telemetry}
    completed: set[str] = set()
    decision_rows, event_rows, artifacts = [], [], {}

    for event in events:
        start, end = event["event_round"], event["end_round"]
        active = event["active_causes_after"]
        canonical = _canonical_bundle(active, manifest, completed)
        rows = []
        for round_idx in range(start, end):
            telemetry_row = by_round[round_idx]
            if not telemetry_row.get("decision_due"):
                continue
            status = str(telemetry_row["decision_bundle_status"])
            if status not in BUNDLE_STATUSES:
                status = "INVALID_SCHEMA"
            actions = _actions(telemetry_row.get("declared_bundle")) if status == "VALID" else ()
            try:
                scored = _score_decision(active, actions, status, manifest, completed)
            except (TypeError, ValueError, KeyError):
                status = "INVALID_SCHEMA"
                actions = ()
                scored = _score_decision(active, actions, status, manifest, completed)

            try:
                observation = json.loads(telemetry_row["public_observation_json"])
            except (TypeError, json.JSONDecodeError):
                return {
                    "integrity": _invalid_integrity(
                        run_id, agent_id, config_id, episode_id,
                        "PUBLIC_OBSERVATION_ARTIFACT_INVALID", round_idx,
                    ), "events": [], "decisions": [], "artifacts": {},
                }
            observation_ref = _artifact(observation, artifacts)
            raw_ref = _artifact({"raw_output": telemetry_row.get("raw_output", "")}, artifacts)
            deadline = event["emit_deadline"]
            decision = {
                "run_id": run_id, "agent_id": agent_id,
                "agent_config_id": config_id, "episode_id": episode_id,
                "event_id": event["event_id"], "decision_round": round_idx,
                "active_causes": _pipe(active),
                "public_observation_ref": observation_ref, "raw_output_ref": raw_ref,
                "parsed_bundle": _pipe(actions), "bundle_status": status,
                "covered_causes": _pipe(scored["covered"]),
                "missing_causes": _pipe(scored["missing"]),
                "unnecessary_actions": _pipe(scored["unnecessary"]),
                "coverage_status": scored["coverage"],
                "decision_label": scored["label"],
                "relative_delay": round_idx - event["first_due_round"],
                "deadline_relation": _deadline_relation(round_idx, deadline),
            }
            decision_rows.append(decision)
            rows.append({**decision, "comparison": scored["comparison"]})

        valid_coverages = [row["coverage_status"] for row in rows
                           if row["coverage_status"]]
        best = (max(valid_coverages, key=COVERAGE_RANK.__getitem__)
                if valid_coverages else "UNASSESSED")
        final = rows[-1]
        final_coverage = final["coverage_status"] or "UNASSESSED"
        correct = [row for row in rows if row["decision_label"] in CORRECT_LABELS]
        ever_correct = bool(correct)
        ended_correct = final["decision_label"] in CORRECT_LABELS
        if ever_correct:
            bucket = "EVER_CORRECT"
        elif best == "FULL":
            bucket = "FULL_WITH_EXTRA_BEST"
        elif best == "PARTIAL":
            bucket = "PARTIAL_BEST"
        elif best == "NONE":
            bucket = "NONE_BEST"
        else:
            bucket = "UNASSESSED"

        bundle_changes, previous_valid = 0, None
        regressions, previous_correct = 0, False
        for row in rows:
            if row["bundle_status"] != "VALID":
                previous_valid = None
            else:
                if previous_valid is not None and row["comparison"] != previous_valid:
                    bundle_changes += 1
                previous_valid = row["comparison"]
            now_correct = row["decision_label"] in CORRECT_LABELS
            if previous_correct and not now_correct:
                regressions += 1
            previous_correct = now_correct

        correct_round = correct[0]["decision_round"] if correct else None
        deadline = event["emit_deadline"]
        timing = ("UNTIMED" if deadline is None else
                  "NOT_REACHED" if correct_round is None else
                  "ON_TIME" if correct_round <= deadline else "LATE")
        unnecessary_seen = sorted({action for row in rows
                                   for action in _actions(row["unnecessary_actions"])})
        over_count = sum(bool(row["unnecessary_actions"]) for row in rows)
        final_valid = final["bundle_status"] == "VALID"
        conclusion = _audit_conclusion(correct_round, deadline, ended_correct,
                                       regressions, over_count)
        event_rows.append({
            "run_id": run_id, "agent_id": agent_id, "agent_config_id": config_id,
            "episode_id": episode_id, "event_id": event["event_id"],
            "case_id": plan["case_id"], "seed": int(plan["training_seed"]),
            "partition_mode": plan["partition_mode"],
            "transition_type": event["transition_type"], "event_round": start,
            "lifecycle_transition_type": _lifecycle_transition_type(
                event["active_causes_before"], active,
            ),
            "first_due_round": event["first_due_round"],
            "next_transition_round": ("" if event["next_transition_round"] is None
                                      else event["next_transition_round"]),
            "active_causes_before": _pipe(event["active_causes_before"]),
            "active_causes_after": _pipe(active),
            "removed_causes": _pipe(event["removed_causes"]),
            "affected_clients": _canonical_json(event["affected_clients"]),
            "severity": _canonical_json(event["severity"]),
            "canonical_bundle": _pipe(canonical),
            "emit_deadline": "" if deadline is None else deadline,
            "correct_decision_round": "" if correct_round is None else correct_round,
            "correct_decision_bundle": (correct[0]["parsed_bundle"] if correct else ""),
            "decision_delay": ("" if correct_round is None else
                               correct_round - event["first_due_round"]),
            "final_bundle_before_transition": (final["parsed_bundle"] if final_valid else ""),
            "final_decision_label": final["decision_label"],
            "decision_count": len(rows), "correct_decision_count": len(correct),
            "invalid_bundle_count": sum(row["bundle_status"] != "VALID" for row in rows),
            "over_intervention_decision_count": over_count,
            "bundle_change_count": bundle_changes,
            "post_correct_regression_count": regressions,
            "best_coverage": best, "final_coverage": final_coverage,
            "event_outcome_bucket": bucket, "ever_correct": ever_correct,
            "ended_correct": ended_correct, "never_correct": not ever_correct,
            "timing_status": timing,
            "missing_causes_final": (final["missing_causes"] if final_valid else ""),
            "unnecessary_actions_seen": _pipe(unnecessary_seen),
            "ever_over_intervened": bool(over_count),
            "final_has_unnecessary_actions": (bool(final["unnecessary_actions"])
                                               if final_valid else ""),
            "audit_conclusion": conclusion,
        })

    integrity = {
        "run_id": run_id, "agent_id": agent_id, "agent_config_id": config_id,
        "episode_id": episode_id, "run_status": "VALID", "failure_type": "",
        "failure_round": "", "rerun_required": False,
    }
    return {"integrity": integrity, "events": event_rows,
            "decisions": decision_rows, "artifacts": artifacts}


def _audit_conclusion(correct_round, deadline, ended_correct: bool,
                      regressions: int, over_count: int) -> str:
    if correct_round is None:
        result = "事件结束前从未给出完整正确决策"
    else:
        result = f"第{correct_round}轮首次完整正确"
        if deadline is not None and correct_round > deadline:
            result += f"，超过截止轮{correct_round - deadline}轮"
    if regressions:
        result += f"；首次正确后回退{regressions}次"
    result += "；最终决策正确" if ended_correct else "；最终决策不正确"
    if over_count:
        result += f"；{over_count}次决策包含多余动作"
    return result + "。"


def _rate(numerator: int, denominator: int):
    return "" if denominator == 0 else round(numerator / denominator, 8)


def summarize_agents(event_rows: list[dict], decision_rows: list[dict],
                     specs: dict[str, dict]) -> list[dict]:
    """Build one micro-summary row per valid Agent configuration."""
    grouped_events = defaultdict(list)
    grouped_decisions = defaultdict(list)
    for row in event_rows:
        grouped_events[row["agent_config_id"]].append(row)
    for row in decision_rows:
        grouped_decisions[row["agent_config_id"]].append(row)

    summaries = []
    for config_id in sorted(grouped_events):
        all_events = grouped_events[config_id]
        events = [row for row in all_events if row["transition_type"] != "INIT"]
        init_events = [row for row in all_events if row["transition_type"] == "INIT"]
        event_keys = {(row["run_id"], row["episode_id"], row["event_id"])
                      for row in events}
        decisions = [row for row in grouped_decisions[config_id]
                     if (row["run_id"], row["episode_id"], row["event_id"]) in event_keys]
        init_keys = {(row["run_id"], row["episode_id"], row["event_id"])
                     for row in init_events}
        init_decisions = [row for row in grouped_decisions[config_id]
                          if (row["run_id"], row["episode_id"], row["event_id"]) in init_keys]
        spec = specs[config_id]
        ever = sum(bool(row["ever_correct"]) for row in events)
        ended = sum(bool(row["ended_correct"]) for row in events)
        timed = [row for row in events if row["emit_deadline"] != ""]
        delays = [int(row["decision_delay"]) for row in events
                  if row["decision_delay"] != ""]
        removals = [row for row in events if row["removed_causes"]]
        timed_removals = [row for row in removals if row["emit_deadline"] != ""]
        removal_delays = [int(row["decision_delay"]) for row in removals
                          if row["decision_delay"] != ""]
        buckets = {name: sum(row["event_outcome_bucket"] == name for row in events)
                   for name in ("EVER_CORRECT", "FULL_WITH_EXTRA_BEST", "PARTIAL_BEST",
                                "NONE_BEST", "UNASSESSED")}
        correct_decisions = sum(row["decision_label"] in CORRECT_LABELS
                                for row in decisions)
        over_decisions = sum(bool(row["unnecessary_actions"]) for row in decisions)
        init_noop = sum(row["decision_label"] == "NO_OP_CORRECT" for row in init_decisions)
        init_over = sum(row["decision_label"] == "OVER_INTERVENTION"
                        for row in init_decisions)
        summaries.append({
            "agent_id": all_events[0]["agent_id"], "agent_config_id": config_id,
            "agent_version": spec.get("agent_version", "n/a"),
            "model": spec.get("model", "n/a"),
            "policy": _canonical_json(spec.get("policy", {
                key: spec[key] for key in ("decision_cadence", "timeout_seconds", "max_calls")
                if key in spec
            })),
            "evaluated_runs": len({(row["run_id"], row["episode_id"])
                                   for row in all_events}),
            "evaluated_events": len(events),
            "ever_correct_event_count": ever,
            "ever_correct_event_rate": _rate(ever, len(events)),
            "ended_correct_event_rate": _rate(ended, len(events)),
            "never_correct_event_count": len(events) - ever,
            "full_with_extra_best_event_count": buckets["FULL_WITH_EXTRA_BEST"],
            "partial_best_event_count": buckets["PARTIAL_BEST"],
            "none_best_event_count": buckets["NONE_BEST"],
            "unassessed_event_count": buckets["UNASSESSED"],
            "correct_decision_rate": _rate(correct_decisions, len(decisions)),
            "invalid_bundle_rate": _rate(
                sum(row["bundle_status"] != "VALID" for row in decisions), len(decisions)
            ),
            "ever_over_intervened_event_rate": _rate(
                sum(bool(row["ever_over_intervened"]) for row in events), len(events)
            ),
            "over_intervention_decision_rate": _rate(over_decisions, len(decisions)),
            "timed_event_count": len(timed),
            "on_time_event_rate": _rate(
                sum(row["timing_status"] == "ON_TIME" for row in timed), len(timed)
            ),
            "median_decision_delay": ("" if not delays else
                                      round(float(statistics.median(delays)), 8)),
            "worst_decision_delay": "" if not delays else max(delays),
            "untimed_event_count": len(events) - len(timed),
            "bundle_change_count": sum(int(row["bundle_change_count"]) for row in events),
            "post_correct_regression_count": sum(
                int(row["post_correct_regression_count"]) for row in events
            ),
            "cause_removal_event_count": len(removals),
            "ever_correct_deescalation_rate": _rate(
                sum(bool(row["ever_correct"]) for row in removals), len(removals)
            ),
            "ended_correct_deescalation_rate": _rate(
                sum(bool(row["ended_correct"]) for row in removals), len(removals)
            ),
            "on_time_deescalation_rate": _rate(
                sum(row["timing_status"] == "ON_TIME" for row in timed_removals),
                len(timed_removals),
            ),
            "never_deescalated_count": sum(not bool(row["ever_correct"])
                                             for row in removals),
            "median_deescalation_delay": ("" if not removal_delays else
                                           round(float(statistics.median(removal_delays)), 8)),
            "init_decision_count": len(init_decisions),
            "init_no_op_correct_rate": _rate(init_noop, len(init_decisions)),
            "init_over_intervention_rate": _rate(init_over, len(init_decisions)),
        })
        if sum(buckets.values()) != len(events):
            raise AssertionError("event outcome buckets do not reconcile")
    return summaries
