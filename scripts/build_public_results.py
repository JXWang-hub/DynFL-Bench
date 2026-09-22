"""Build and verify the small, redacted paper-evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_results"
MAIN_MANIFEST_SHA = "af5afd16150f4378edeace0ad00151904c7fb8c6d15e8bcebb9ee9a8e3eb540d"
ATOMIC_MANIFEST_SHA = "c1210bb72c632e08764a96d310d1aadf44ed3ee7f5417931db77a97433f4da9d"
PROMPT_SHA = "01779b937baf079ebc7b9c37a757e70bc0f40f83e784170b15850fa54f5ae291"

AGENTS = {
    "random_bundle_v2": "Random bundle",
    "composite_rule": "Composite Rule",
    "qwen3p8_flash_none": "Qwen none",
    "qwen3p8_flash_low": "Qwen low",
    "qwen3p8_flash_xhigh": "Qwen xhigh",
    "glm_5p2_low": "GLM low",
    "deepseek_v4_flash_0731_low": "DeepSeek low",
    "noop": "NoOp",
    "stress_oracle": "Stress Oracle",
    "temporal_oracle": "Temporal Oracle",
}
PAPER_AGENTS = set(AGENTS) - {"noop", "stress_oracle", "temporal_oracle"}
ARCHIVE_DISPLAY_TO_ID = {
    "Random bundle v2": "random_bundle_v2",
    "Composite Rule": "composite_rule",
    "Qwen3.8-Flash (none)": "qwen3p8_flash_none",
    "Qwen3.8-Flash (low)": "qwen3p8_flash_low",
    "Qwen3.8-Flash (xhigh)": "qwen3p8_flash_xhigh",
    "GLM-5.2 (low)": "glm_5p2_low",
    "DeepSeek-V4-Flash (low)": "deepseek_v4_flash_0731_low",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(value: str) -> tuple[int | None, int | None]:
    if not value or "/" not in value or value.lower() == "n/a":
        return None, None
    left, right = value.split("/", 1)
    return int(left), int(right)


def percentage(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}"


def source_record(archive: Path, relative: str) -> dict[str, object]:
    path = archive / Path(relative)
    return {
        "id": f"archive:{relative.replace('\\\\', '/')}",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def ablation_source_root(archive: Path) -> Path:
    matches = list((archive / "source_audits").glob("*/g1_qwen/per_case.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one observation-ablation source, found {len(matches)}")
    return matches[0].parents[1]


def build_core(archive: Path) -> tuple[int, int]:
    fields = [
        "scenario_set", "case", "seed", "agent", "completed", "ever_exact",
        "teh_at_5", "teh_at_10", "ended_correct", "post_hit_retention_percent",
        "first_regression_delay_rounds", "regressions", "evidence_status",
    ]
    rows: list[dict[str, object]] = []
    single = read_csv(archive / "single_7/events_consolidated.csv")
    for row in single:
        if row["agent"] not in PAPER_AGENTS:
            continue
        rows.append({
            "scenario_set": "single_cause", "case": row["case"], "seed": row["seed"],
            "agent": AGENTS[row["agent"]], "completed": row["completed"],
            "ever_exact": row["ever_exact"], "teh_at_5": row["success5"],
            "teh_at_10": row["success10"], "ended_correct": row["ended_correct"],
            "post_hit_retention_percent": "", "first_regression_delay_rounds": "",
            "regressions": "", "evidence_status": "event_audit",
        })

    two = read_csv(archive / "composite_9/post_hit_metrics_by_run.csv")
    for row in two:
        agent_id = ARCHIVE_DISPLAY_TO_ID.get(row["agent"])
        if agent_id not in PAPER_AGENTS:
            continue
        rows.append({
            "scenario_set": "two_cause", "case": row["case"], "seed": row["seed"],
            "agent": AGENTS[agent_id], "completed": row["completed"],
            "ever_exact": row["ever_exact"], "teh_at_5": row["timely_exact_hit@5"],
            "teh_at_10": row["timely_exact_hit@10"], "ended_correct": row["ended_correct"],
            "post_hit_retention_percent": row["post_hit_retention_percent"],
            "first_regression_delay_rounds": row["first_regression_delay_rounds"],
            "regressions": row["correct_to_incorrect_transitions"],
            "evidence_status": row["status"].lower(),
        })
    rows.sort(key=lambda row: (str(row["scenario_set"]), str(row["case"]), str(row["agent"]), int(row["seed"])))
    write_csv(OUT / "core_episode_results.csv", rows, fields)
    single_count = sum(row["scenario_set"] == "single_cause" for row in rows)
    two_count = sum(row["scenario_set"] == "two_cause" for row in rows)
    assert (single_count, two_count) == (105, 189)
    return single_count, two_count


def metric_row(protocol: str, phase: str, agent: str, metric: str, value: object,
               numerator: object = "", denominator: object = "", unit: str = "percent",
               source: str = "") -> dict[str, object]:
    return {
        "protocol": protocol, "phase": phase, "agent": agent, "metric": metric,
        "numerator": numerator, "denominator": denominator, "value": value,
        "unit": unit, "source_artifact": source,
    }


def build_temporal(archive: Path) -> int:
    rows: list[dict[str, object]] = []
    sequential_source = "archive:sequential_injection/metrics_summary.csv"
    for row in read_csv(archive / "sequential_injection/metrics_summary.csv"):
        agent = AGENTS.get(row["agent"], row["agent"])
        for column, metric in (
            ("ever_exact", "ever_exact"), ("success@10", "teh_at_10"),
            ("ended_correct", "ended_correct"),
            ("runs_with_over_intervention", "episodes_with_over_intervention"),
        ):
            num, den = fraction(row[column])
            rows.append(metric_row("sequential_addition", row["phase"], agent, metric,
                                   percentage(num, den), num, den, source=sequential_source))
        rows.append(metric_row("sequential_addition", row["phase"], agent,
                               "post_hit_regressions", row["post_correct_regressions"],
                               unit="count", source=sequential_source))

    lifecycle_root = archive / "lifecycle_120/b3_composite_v2/noniid"
    lifecycle_specs = [
        ("three_cause_lifecycle", "lifecycle_3cause_qwen_low_t06"),
        ("recurrence", "lifecycle_recurrence_qwen_low_t06"),
    ]
    for protocol, dirname in lifecycle_specs:
        base = lifecycle_root / dirname / "private/agent_decision_audit"
        source = f"archive:lifecycle_120/b3_composite_v2/noniid/{dirname}/private/agent_decision_audit/agent_decision_summary.csv"
        for row in read_csv(base / "agent_decision_summary.csv"):
            if row["agent_id"] not in {"composite_rule", "qwen3p8_flash_low"}:
                continue
            agent = AGENTS[row["agent_id"]]
            events = int(row["evaluated_events"])
            ever = int(row["ever_correct_event_count"])
            ended = round(float(row["ended_correct_event_rate"]) * events)
            rows.extend([
                metric_row(protocol, "all_intervals", agent, "ever_exact", percentage(ever, events), ever, events, source=source),
                metric_row(protocol, "all_intervals", agent, "ended_correct", percentage(ended, events), ended, events, source=source),
                metric_row(protocol, "all_intervals", agent, "roundwise_exact", f"{100 * float(row['correct_decision_rate']):.1f}", source=source),
                metric_row(protocol, "all_intervals", agent, "over_intervention", f"{100 * float(row['over_intervention_decision_rate']):.1f}", source=source),
                metric_row(protocol, "all_intervals", agent, "post_hit_regressions", row["post_correct_regression_count"], unit="count", source=source),
            ])
        event_source = source.replace("agent_decision_summary.csv", "event_decision_audit.csv")
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(base / "event_decision_audit.csv"):
            if row["agent_id"] in {"composite_rule", "qwen3p8_flash_low"} and row["transition_type"] != "INIT":
                grouped[(row["agent_id"], row["transition_type"])].append(row)
        phase_names = {"ACTIVATE": "activation", "CHANGE": "partial_removal", "STABLE": "full_clear"}
        for (agent_id, transition), selected in grouped.items():
            ever = sum(row["ever_correct"] == "True" for row in selected)
            ended = sum(row["ended_correct"] == "True" for row in selected)
            den = len(selected)
            phase = phase_names[transition]
            rows.append(metric_row(protocol, phase, AGENTS[agent_id], "ever_exact", percentage(ever, den), ever, den, source=event_source))
            rows.append(metric_row(protocol, phase, AGENTS[agent_id], "ended_correct", percentage(ended, den), ended, den, source=event_source))

    rotation_source = "archive:phase6_rotation/agent_decision_summary.csv"
    for row in read_csv(archive / "phase6_rotation/agent_decision_summary.csv"):
        if row["agent_id"] not in {"composite_rule", "qwen3p8_flash_low"}:
            continue
        agent = AGENTS[row["agent_id"]]
        events = int(row["evaluated_events"])
        ever = int(row["ever_correct_event_count"])
        ended = round(float(row["ended_correct_event_rate"]) * events)
        rows.extend([
            metric_row("rotation", "all_intervals", agent, "ever_exact", percentage(ever, events), ever, events, source=rotation_source),
            metric_row("rotation", "all_intervals", agent, "ended_correct", percentage(ended, events), ended, events, source=rotation_source),
            metric_row("rotation", "all_intervals", agent, "roundwise_exact", f"{100 * float(row['correct_decision_rate']):.1f}", source=rotation_source),
            metric_row("rotation", "all_intervals", agent, "over_intervention", f"{100 * float(row['over_intervention_decision_rate']):.1f}", source=rotation_source),
            metric_row("rotation", "all_intervals", agent, "post_hit_regressions", row["post_correct_regression_count"], unit="count", source=rotation_source),
        ])
    write_csv(OUT / "temporal_results.csv", rows)
    return len(rows)


def build_ablation(archive: Path) -> int:
    rows: list[dict[str, object]] = []
    cases = {
        "G1": ["atomic_fault", "atomic_real"],
        "G2": ["atomic_fault", "staggered"],
        "G3": ["atomic_label", "atomic_virtual"],
        "G4": ["atomic_dropout", "atomic_hetero"],
    }
    single = read_csv(archive / "single_7/metrics_by_case_agent.csv")
    for group, selected_cases in cases.items():
        for agent, source_names in (
            ("Composite Rule", [f"{group.lower()}_rule"]),
            ("Qwen low", [f"{group.lower()}_qwen"]),
        ):
            full_rows = [row for row in single if row["case"] in selected_cases and row["agent"].startswith(agent.split()[0])]
            full_num = sum(fraction(row["success@10"])[0] for row in full_rows)
            full_den = sum(fraction(row["success@10"])[1] for row in full_rows)
            rows.append({"group": group, "agent": agent, "metric": "teh_at_10", "setting": "full",
                         "numerator": full_num, "denominator": full_den, "value": percentage(full_num, full_den),
                         "unit": "percent", "source_artifact": "archive:single_7/metrics_by_case_agent.csv"})
            masked_rows: list[dict[str, str]] = []
            root = ablation_source_root(archive)
            if group == "G2":
                suffix = "rule" if agent == "Composite Rule" else "qwen"
                source_names = [f"g2_{suffix}_fault", f"g2_{suffix}_staggered"]
            for source_name in source_names:
                masked_rows.extend(read_csv(root / source_name / "per_case.csv"))
            masked_num = sum(int(row["success10"]) for row in masked_rows)
            masked_den = sum(int(row["planned"]) for row in masked_rows)
            source_ids = ";".join(f"archive:observation_ablation/{name}/per_case.csv" for name in source_names)
            rows.append({"group": group, "agent": agent, "metric": "teh_at_10", "setting": "masked",
                         "numerator": masked_num, "denominator": masked_den, "value": percentage(masked_num, masked_den),
                         "unit": "percent", "source_artifact": source_ids})

    g5_source = "archive:g5_retention/summary.csv"
    for row in read_csv(archive / "g5_retention/summary.csv"):
        if row["scope"] != "All":
            continue
        for column, metric, unit in (
            ("Ever", "ever_exact", "percent"), ("TEH10", "teh_at_10", "percent"),
            ("End", "ended_correct", "percent"), ("PHR", "post_hit_retention", "percent"),
            ("FRD", "first_regression_delay", "rounds"),
            ("Hit_to_end", "hit_to_end_without_regression", "percent"),
            ("Regr", "regressions", "count"),
        ):
            num, den = fraction(row[column])
            value = percentage(num, den) if num is not None and metric != "regressions" else row[column]
            rows.append({"group": "G5", "agent": row["controller"], "metric": metric,
                         "setting": row["setting"].lower(), "numerator": "" if num is None else num,
                         "denominator": "" if den is None else den, "value": value, "unit": unit,
                         "source_artifact": g5_source})
    write_csv(OUT / "observation_ablation.csv", rows)
    return len(rows)


def build_speech(archive: Path) -> int:
    rows: list[dict[str, object]] = []
    speech = archive / "supplemental/speech_commands"
    episodes: list[tuple[list[dict[str, str]], str]] = []
    for dirname, action in (("dropout_random", "dropout_handle"), ("virtual_drift", "moment_align_adapt")):
        traces = [row for row in read_csv(speech / dirname / "decision_trace.csv") if row["agent"] == "llm"]
        for seed in ("0", "44", "56"):
            episodes.append(([row for row in traces if row["seed"] == seed], action))
    pre_total = pre_noop = timely = ended = clean = 0
    for episode, action in episodes:
        episode.sort(key=lambda row: int(row["round"]))
        pre = [row for row in episode if int(row["round"]) < 40]
        post = [row for row in episode if int(row["round"]) >= 40]
        pre_total += len(pre)
        pre_noop += sum(row["action"] == "no_op" for row in pre)
        timely += any(row["action"] == action for row in post[:10])
        ended += post[-1]["action"] == action
        clean += all(row["action"] == action for row in post[:10])
    decision_source = "archive:supplemental/speech_commands/{dropout_random,virtual_drift}/decision_trace.csv"
    for metric, num, den in (
        ("pre_event_no_op", pre_noop, pre_total), ("teh_at_10", timely, len(episodes)),
        ("ended_correct", ended, len(episodes)), ("clean_window_at_10", clean, len(episodes)),
    ):
        rows.append({"section": "decision_audit", "scenario": "dropout_and_virtual_drift",
                     "controller": "Qwen low", "measure": metric, "numerator": num,
                     "denominator": den, "value": percentage(num, den), "unit": "percent",
                     "source_artifact": decision_source})

    metrics = {row["agent"]: row for row in read_csv(speech / "virtual_drift/metrics_all_multiseed.csv")}
    noop, rule = metrics["noop"], metrics["composite_rule"]
    source = "archive:supplemental/speech_commands/virtual_drift/metrics_all_multiseed.csv"
    performance = [
        ("post_drift_mean_accuracy_gain", 100 * (float(rule["post_drift_mean_acc_mean"]) - float(noop["post_drift_mean_acc_mean"])), "percentage_points"),
        ("worst_class_accuracy_gain", 100 * (float(rule["post_event_worst_acc_mean"]) - float(noop["post_event_worst_acc_mean"])), "percentage_points"),
        ("recovery_rounds_aligned", float(rule["recovery_rounds_mean"]), "rounds"),
        ("recovery_rounds_noop", float(noop["recovery_rounds_mean"]), "rounds"),
    ]
    for measure, value, unit in performance:
        rows.append({"section": "intervention_value", "scenario": "virtual_drift",
                     "controller": "Composite Rule vs NoOp", "measure": measure,
                     "numerator": "", "denominator": "", "value": f"{value:.2f}",
                     "unit": unit, "source_artifact": source})
    write_csv(OUT / "speech_results.csv", rows)
    return len(rows)


def build_mechanism(archive: Path) -> int:
    rows = read_csv(archive / "offline_figures/action_mechanism.csv")
    write_csv(OUT / "mechanism_validation.csv", rows)
    return len(rows)


def build_provenance(archive: Path, counts: dict[str, int]) -> None:
    source_groups = {
        "core_results": [
            "single_7/events_consolidated.csv", "single_7/rounds_consolidated.csv",
            "single_7/stable_rounds_consolidated.csv", "composite_9/events_consolidated.csv",
            "composite_9/rounds_consolidated.csv", "composite_9/stable_rounds_consolidated.csv",
            "composite_9/post_hit_metrics_by_run.csv",
        ],
        "temporal_results": [
            "sequential_injection/metrics_summary.csv",
            "lifecycle_120/b3_composite_v2/noniid/lifecycle_3cause_qwen_low_t06/private/agent_decision_audit/agent_decision_summary.csv",
            "lifecycle_120/b3_composite_v2/noniid/lifecycle_3cause_qwen_low_t06/private/agent_decision_audit/event_decision_audit.csv",
            "lifecycle_120/b3_composite_v2/noniid/lifecycle_recurrence_qwen_low_t06/private/agent_decision_audit/agent_decision_summary.csv",
            "lifecycle_120/b3_composite_v2/noniid/lifecycle_recurrence_qwen_low_t06/private/agent_decision_audit/event_decision_audit.csv",
            "phase6_rotation/agent_decision_summary.csv",
        ],
        "observation_ablation": [
            "single_7/metrics_by_case_agent.csv", "g5_retention/summary.csv",
        ],
        "speech_results": [
            "supplemental/speech_commands/dropout_random/decision_trace.csv",
            "supplemental/speech_commands/virtual_drift/decision_trace.csv",
            "supplemental/speech_commands/virtual_drift/metrics_all_multiseed.csv",
        ],
        "mechanism_validation": ["offline_figures/action_mechanism.csv"],
    }
    provenance = {
        "schema_version": 1,
        "release_state": "public_release",
        "paper_version": "ICASSP 2027 manuscript revision dated 2026-09-22",
        "frozen_identifiers": {
            "main_manifest_sha256": MAIN_MANIFEST_SHA,
            "atomic_addendum_manifest_sha256": ATOMIC_MANIFEST_SHA,
            "prompt_sha256": PROMPT_SHA,
            "evaluation_seeds": [0, 44, 56],
            "lifecycle_runner_commit": "3a6a7532f2dfea64fd164716cc3c34c9c7e83318",
            "single_random_runner_commit": "ad9efca24ba29b243f6516db9225af6657d76843",
        },
        "paper_to_artifact": {
            "core_result_table": ["summary.csv", "core_episode_results.csv"],
            "temporal_protocol_claims": ["temporal_results.csv"],
            "observation_ablation_table_and_g5_retention": ["observation_ablation.csv"],
            "speech_commands_table": ["speech_results.csv"],
            "cause_action_mechanism_validation": ["mechanism_validation.csv"],
            "trajectory_retention_and_regression": ["core_episode_results.csv"],
        },
        "source_artifacts": {
            name: [source_record(archive, relative) for relative in relatives]
            for name, relatives in source_groups.items()
        },
        "data_classification": {
            "core_and_temporal": "real CIFAR-10 runs plus frozen API decisions",
            "speech": "real Speech Commands runs plus frozen API decisions",
            "mechanism_validation": "calibration-seed intervention comparisons",
            "api_policy": "frozen outputs reused; raw request/response payloads excluded",
        },
        "metric_denominators": {
            "exact": "all planned event decision slots; slots lost to terminal failure count as incorrect",
            "over_intervention": "observed scorable event decisions",
            "stable_false_intervention": "observed stable-period decisions",
            "post_hit_retention": "macro-average over intervals that reach an exact decision",
        },
        "export_counts": counts,
        "excluded": [
            "datasets", "model checkpoints", "raw API payloads", "credentials",
            "absolute paths", "private hostnames", "temporary and superseded runs",
        ],
    }
    ablation_sources = []
    root = ablation_source_root(archive)
    for name in (
        "g1_qwen", "g1_rule", "g2_qwen_fault", "g2_qwen_staggered",
        "g2_rule_fault", "g2_rule_staggered", "g3_qwen", "g3_rule",
        "g4_qwen", "g4_rule",
    ):
        path = root / name / "per_case.csv"
        ablation_sources.append({
            "id": f"archive:observation_ablation/{name}/per_case.csv",
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    provenance["source_artifacts"]["observation_ablation"].extend(ablation_sources)
    with (OUT / "provenance.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")


def write_integrity(counts: dict[str, int]) -> None:
    summary = read_csv(OUT / "summary.csv")
    lookup = {(row["scenario"], row["agent"]): row for row in summary}
    assertions = {
        "summary_has_12_rows": len(summary) == 12,
        "single_random_matches_paper": lookup[("single_cause", "Random bundle")]["teh_at_10"] == "4.8",
        "two_cause_random_oi_uses_observed_denominator": lookup[("two_cause", "Random bundle")]["oi"] == "7.2",
        "two_cause_qwen_low_matches_paper": lookup[("two_cause", "Qwen low")]["phr"] == "54.0",
        "episode_counts_match_protocol": counts["core_single_episode_rows"] == 105 and counts["core_two_cause_episode_rows"] == 189,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    integrity = {
        "schema_version": 1,
        "status": "PASS",
        "source_archive_verified": True,
        "generated_files_verified": True,
        "assertions": assertions,
        "counts": counts,
    }
    with (OUT / "run_integrity.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(integrity, indent=2) + "\n")


def write_checksums() -> None:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.csv":
            rows.append({"relative_path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(OUT / "SHA256SUMS.csv", rows)


def verify() -> None:
    rows = read_csv(OUT / "SHA256SUMS.csv")
    failures = []
    for row in rows:
        path = OUT / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            failures.append(row["relative_path"])
    if failures:
        raise SystemExit(f"checksum verification failed: {failures}")
    integrity = json.loads((OUT / "run_integrity.json").read_text(encoding="utf-8"))
    if integrity["status"] != "PASS" or not all(integrity["assertions"].values()):
        raise SystemExit("integrity assertions failed")
    print(f"PUBLIC_RESULTS_VERIFY_PASS files={len(rows)}")


def build(archive: Path) -> None:
    if not (archive / "SHA256SUMS.csv").is_file():
        raise SystemExit(f"not a final result archive: {archive}")
    single_count, two_count = build_core(archive)
    counts = {
        "core_single_episode_rows": single_count,
        "core_two_cause_episode_rows": two_count,
        "temporal_metric_rows": build_temporal(archive),
        "observation_ablation_rows": build_ablation(archive),
        "speech_metric_rows": build_speech(archive),
        "mechanism_validation_rows": build_mechanism(archive),
    }
    build_provenance(archive, counts)
    write_integrity(counts)
    write_checksums()
    verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="path to the private final result archive")
    parser.add_argument("--verify", action="store_true", help="verify the existing public package")
    args = parser.parse_args()
    if args.archive:
        build(args.archive.resolve())
    elif args.verify:
        verify()
    else:
        parser.error("provide --archive to build or --verify to verify")


if __name__ == "__main__":
    main()
