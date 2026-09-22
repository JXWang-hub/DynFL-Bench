"""Pure artifact auditor for frozen B3 plans, raw calibration rows, and matrices."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from b3_manifest import (
    DEFAULT_MANIFEST, load_manifest, manifest_sha256, validate_manifest,
    require_formal_manifest,
)
from b3_fingerprint import source_sha256
from evidence import ACTION_MECHANISM_SIGNAL


ROOT = Path(__file__).resolve().parent
CAUSE_CONFIG = {
    "real_drift": ("drift", "real"),
    "virtual_drift": ("drift", "virtual"),
    "label_prior_drift": ("drift", "label"),
    "fault": ("fault", None),
    "dropout": ("dropout", None),
    "partial_participation_hetero": ("hetero", None),
    "staggered_concepts": ("staggered", None),
}
FEDDRIFT_FIELDS = (
    "feddrift_ari", "feddrift_nmi", "feddrift_learner_count_error",
    "feddrift_trigger_delay", "feddrift_assignment_churn",
    "feddrift_split_count", "feddrift_merge_count",
)
CLEAN_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def matrix_sha256(matrix: dict) -> str:
    payload = copy.deepcopy(matrix)
    payload.pop("matrix_sha256", None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit_manifest(manifest: dict, root: Path = ROOT) -> str:
    errors = validate_manifest(manifest, root)
    _require(not errors, "invalid manifest: " + "; ".join(errors))
    calibration = set(manifest["calibration"]["seeds"])
    evaluation = set(manifest["protocol"]["evaluation_seeds"])
    _require(calibration.isdisjoint(evaluation), "calibration/evaluation seed leak")
    return manifest_sha256(manifest)


def _expected_jobs(manifest: dict, partition_mode: str = "noniid") -> list[dict]:
    calibration = manifest["calibration"]
    suite_sha = manifest_sha256(manifest)
    jobs = []
    for cause in calibration["causes"]:
        scenario, drift_type = CAUSE_CONFIG[cause]
        for seed in calibration["seeds"]:
            for action in calibration["actions"]:
                identity = f"{suite_sha}:cifar10:{partition_mode}:0:{cause}:{seed}:{action}"
                jobs.append({
                    "run_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    "cause": cause,
                    "scenario": scenario,
                    "drift_type": drift_type,
                    "seed": seed,
                    "action": action,
                })
    return jobs


def audit_calibration_plan(plan: dict, manifest: dict) -> list[dict]:
    partition_mode = plan.get("partition_mode")
    _require(partition_mode in manifest["protocol"]["partition_modes"],
             "calibration plan partition mode mismatch")
    expected = _expected_jobs(manifest, partition_mode)
    _require(plan.get("schema_version") == manifest["schema_version"],
             "calibration plan schema mismatch")
    _require(plan.get("manifest_sha256") == manifest_sha256(manifest),
             "calibration plan manifest SHA mismatch")
    _require(plan.get("dataset") == "cifar10" and plan.get("smoke") is False,
             "formal calibration plan must use CIFAR10 with smoke=false")
    jobs = plan.get("jobs")
    _require(isinstance(jobs, list), "calibration plan jobs must be a list")
    _require(plan.get("job_count") == len(jobs) == len(expected),
             f"formal calibration plan requires {len(expected)} exact jobs")
    _require(len({job.get("run_id") for job in jobs}) == len(jobs),
             "calibration plan contains duplicate run_id values")
    fields = ("run_id", "cause", "scenario", "drift_type", "seed", "action")
    _require(
        [tuple(job.get(field) for field in fields) for job in jobs]
        == [tuple(job[field] for field in fields) for job in expected],
        "calibration plan job grid/order/run_id mismatch",
    )
    return expected


def _median_metric(rows: list[dict], field: str):
    values = []
    for row in rows:
        value = row.get(field, "n/a")
        if value in (None, "", "n/a"):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"raw calibration has invalid {field}") from exc
        _require(math.isfinite(number), f"raw calibration has invalid {field}")
        if field not in {"feddrift_ari", "feddrift_nmi"}:
            _require(number >= 0, f"raw calibration has invalid {field}")
        values.append(number)
    return round(statistics.median(values), 8) if values else "n/a"


def aggregate_cause_rows(manifest: dict, rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        try:
            utility = float(row["post_event_mean_accuracy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw calibration has invalid utility") from exc
        _require(math.isfinite(utility), "raw calibration utility is not finite")
        grouped[(row["cause"], row["requested_action"])].append(utility)

    calibration = manifest["calibration"]
    epsilon = float(calibration["epsilon"])
    causes = {}
    for cause in calibration["causes"]:
        utilities = {
            action: statistics.median(grouped[(cause, action)])
            for action in calibration["actions"]
        }
        feasible = set(manifest["causes"][cause]["feasible_actions"])
        u_star = max(utilities[action] for action in feasible)
        u_noop = utilities["no_op"]
        raw_denominator = u_star - u_noop
        denominator = max(raw_denominator, epsilon)
        informative = raw_denominator >= epsilon
        actions = {}
        for action, utility in utilities.items():
            relative_damage = (u_star - utility) / denominator
            if action in feasible:
                penalty = 0.0
            elif action == "no_op":
                penalty = float(calibration["missed_cause_penalty"])
            elif not informative:
                penalty = float(calibration["uninformative_penalty"])
            else:
                penalty = min(
                    float(calibration["wrong_action_penalty_ceiling"]),
                    max(float(calibration["wrong_action_penalty_floor"]),
                        0.5 * relative_damage),
                )
            actions[action] = {
                "median_utility": round(utility, 8),
                "relative_damage": round(relative_damage, 8),
                "penalty": round(penalty, 8),
                "feasible": action in feasible,
                "evidence": manifest["actions"][action]["evidence"],
            }
        causes[cause] = {
            "u_star": round(u_star, 8),
            "u_noop": round(u_noop, 8),
            "raw_denominator": round(raw_denominator, 8),
            "denominator": round(denominator, 8),
            "informative": informative,
            "actions": actions,
        }
        if cause == "staggered_concepts":
            causes[cause]["structure"] = {
                action: {
                    f"median_{field}": _median_metric([
                        row for row in rows
                        if row["cause"] == cause and row["requested_action"] == action
                    ], field)
                    for field in FEDDRIFT_FIELDS
                }
                for action in calibration["actions"]
            }
    return causes


def _expected_cause_rows(manifest: dict, rows: list[dict]) -> dict:
    return aggregate_cause_rows(manifest, rows)


def audit_calibration(manifest: dict, plan: dict, raw_path: Path, matrix: dict,
                      root: Path = ROOT, expected_code_sha: str | None = None) -> dict:
    suite_sha = audit_manifest(manifest, root)
    expected_jobs = audit_calibration_plan(plan, manifest)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    required = {
        "run_id", "manifest_sha256", "code_sha", "source_sha256", "git_dirty",
        "git_status_sha256", "dataset", "partition_mode", "smoke", "cause", "scenario",
        "drift_type", "seed", "requested_action", "effective_tool",
        "action_emitted", "activation_observed", "mechanism_observed",
        "mechanism_signal", "post_event_mean_accuracy", *FEDDRIFT_FIELDS,
    }
    _require(required <= fields, "raw calibration is missing required columns")
    ids = [row["run_id"] for row in rows]
    _require(len(ids) == len(set(ids)), "raw calibration contains duplicate run_id values")
    expected_by_id = {job["run_id"]: job for job in expected_jobs}
    _require(len(rows) == len(expected_jobs) and set(ids) == set(expected_by_id),
             f"raw calibration requires {len(expected_jobs)} exact jobs")

    for row in rows:
        job = expected_by_id[row["run_id"]]
        expected_drift = job["drift_type"] or "n/a"
        _require(
            row["manifest_sha256"] == suite_sha
            and row["dataset"] == "cifar10"
            and row["partition_mode"] == plan["partition_mode"]
            and row["smoke"] == "false"
            and row["cause"] == job["cause"]
            and row["scenario"] == job["scenario"]
            and row["drift_type"] == expected_drift
            and row["seed"] == str(job["seed"])
            and row["requested_action"] == job["action"],
            f"raw calibration metadata mismatch for {row['run_id']}",
        )
        _require(row["git_dirty"] == "false", "formal calibration used dirty code")
        _require(row["git_status_sha256"] == CLEAN_STATUS_SHA256,
                 "formal calibration git status hash is not clean")
        _require(len(row["code_sha"]) == 40 and all(c in "0123456789abcdef" for c in row["code_sha"]),
                 "raw calibration has invalid code SHA")
        _require(row["source_sha256"] == source_sha256(),
                 "raw calibration source fingerprint mismatch")
        action = row["requested_action"]
        emitted = row["action_emitted"] == "true"
        activated = row["activation_observed"] == "true"
        mechanism = row["mechanism_observed"] == "true"
        _require(row["action_emitted"] in {"true", "false"}
                 and row["activation_observed"] in {"true", "false"}
                 and row["mechanism_observed"] in {"true", "false"},
                 "raw calibration has invalid mechanism boolean")
        _require(row["mechanism_signal"] == ACTION_MECHANISM_SIGNAL[action],
                 "raw calibration mechanism signal mismatch")
        _require(not activated or emitted, "activated action was not emitted")
        _require(not mechanism or activated, "mechanism observed without activation")
        expected_tool = "none" if action == "no_op" or not activated else action
        _require(row["effective_tool"] == expected_tool,
                 "raw calibration effective tool mismatch")
        if action in manifest["causes"][job["cause"]]["feasible_actions"]:
            _require(emitted and activated and mechanism,
                     "feasible action was not emitted/activated/observed")
        if not activated:
            _require(not mechanism, "requested-but-no-op action claims a mechanism")
        if job["cause"] == "staggered_concepts":
            for field in FEDDRIFT_FIELDS:
                _median_metric([row], field)

    code_shas = sorted({row["code_sha"] for row in rows})
    source_shas = sorted({row["source_sha256"] for row in rows})
    _require(len(code_shas) == 1, "formal calibration must use one code SHA")
    _require(source_shas == [source_sha256()],
             "formal calibration must use the current source fingerprint")
    if expected_code_sha is not None:
        _require(code_shas == [expected_code_sha], "calibration used the wrong candidate code SHA")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    calibration = manifest["calibration"]
    _require(matrix.get("schema_version") == manifest["schema_version"]
             and matrix.get("status") == "frozen",
             "damage matrix version/fingerprint mismatch")
    _require(matrix.get("manifest_sha256") == suite_sha, "damage matrix manifest SHA mismatch")
    _require(matrix.get("source_raw_file") == raw_path.name, "damage matrix raw filename mismatch")
    _require(matrix.get("source_raw_sha256") == raw_sha, "damage matrix raw SHA mismatch")
    _require(matrix.get("code_shas") == code_shas, "damage matrix code SHA mismatch")
    _require(matrix.get("source_sha256") == source_shas[0],
             "damage matrix source fingerprint mismatch")
    _require(matrix.get("git_dirty_any") is False, "damage matrix contains dirty runs")
    _require(matrix.get("dataset") == "cifar10", "damage matrix dataset mismatch")
    _require(matrix.get("partition_mode") == plan["partition_mode"],
             "damage matrix partition mode mismatch")
    _require(matrix.get("seeds") == calibration["seeds"], "damage matrix seed mismatch")
    _require(matrix.get("primary_utility") == calibration["primary_utility"],
             "damage matrix utility mismatch")
    _require(matrix.get("epsilon") == calibration["epsilon"], "damage matrix epsilon mismatch")
    _require(matrix.get("run_count") == len(expected_jobs),
             "damage matrix run_count mismatch")
    _require(matrix.get("causes") == _expected_cause_rows(manifest, rows),
             "damage matrix values do not reproduce from raw rows")
    _require(matrix.get("matrix_sha256") == matrix_sha256(matrix),
             "damage matrix self SHA mismatch")
    return {"jobs": len(rows), "manifest_sha256": suite_sha,
            "code_sha": code_shas[0], "matrix_sha256": matrix["matrix_sha256"]}


def _fixture_plan(manifest: dict, partition_mode: str = "noniid") -> dict:
    jobs = _expected_jobs(manifest, partition_mode)
    return {"schema_version": manifest["schema_version"],
            "manifest_sha256": manifest_sha256(manifest),
            "dataset": "cifar10", "partition_mode": partition_mode,
            "smoke": False, "job_count": len(jobs), "jobs": jobs}


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_rows(plan: dict) -> list[dict]:
    code_sha = "1" * 40
    rows = []
    for index, job in enumerate(plan["jobs"]):
        rows.append({
            "run_id": job["run_id"],
            "manifest_sha256": plan["manifest_sha256"],
            "code_sha": code_sha,
            "source_sha256": source_sha256(),
            "git_dirty": "false",
            "git_status_sha256": CLEAN_STATUS_SHA256,
            "dataset": "cifar10",
            "partition_mode": plan["partition_mode"],
            "smoke": "false",
            "cause": job["cause"],
            "scenario": job["scenario"],
            "drift_type": job["drift_type"] or "n/a",
            "seed": str(job["seed"]),
            "requested_action": job["action"],
            "effective_tool": "none" if job["action"] == "no_op" else job["action"],
            "action_emitted": "true",
            "activation_observed": "true",
            "mechanism_observed": "true",
            "mechanism_signal": ACTION_MECHANISM_SIGNAL[job["action"]],
            "post_event_mean_accuracy": f"{0.5 + (index % 9) * 0.01:.8f}",
            **{field: ("0.5" if job["cause"] == "staggered_concepts" else "n/a")
               for field in FEDDRIFT_FIELDS},
        })
    return rows


def _fixture_matrix(manifest: dict, rows: list[dict], raw_path: Path) -> dict:
    calibration = manifest["calibration"]
    matrix = {
        "schema_version": manifest["schema_version"],
        "status": "frozen",
        "manifest_sha256": manifest_sha256(manifest),
        "source_raw_file": raw_path.name,
        "source_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "code_shas": [rows[0]["code_sha"]],
        "source_sha256": rows[0]["source_sha256"],
        "git_dirty_any": False,
        "dataset": "cifar10",
        "partition_mode": rows[0]["partition_mode"],
        "seeds": calibration["seeds"],
        "primary_utility": calibration["primary_utility"],
        "epsilon": calibration["epsilon"],
        "run_count": len(rows),
        "causes": _expected_cause_rows(manifest, rows),
    }
    matrix["matrix_sha256"] = matrix_sha256(matrix)
    return matrix


def _expect_invalid(fn, text: str) -> None:
    try:
        fn()
    except ValueError as exc:
        _require(text in str(exc).lower(), f"unexpected negative-audit error: {exc}")
    else:
        raise AssertionError(f"negative audit did not fail: {text}")


def self_check(manifest_path: Path = DEFAULT_MANIFEST) -> None:
    if not __debug__:
        raise RuntimeError("artifact-audit self-check refuses optimized mode")
    manifest = load_manifest(manifest_path)
    plan = _fixture_plan(manifest)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_path = root / "atomic_action_calibration_raw.csv"
        rows = _fixture_rows(plan)
        _write_rows(raw_path, rows)
        matrix = _fixture_matrix(manifest, rows, raw_path)
        result = audit_calibration(manifest, plan, raw_path, matrix,
                                   expected_code_sha="1" * 40)

        bad_hash = copy.deepcopy(matrix)
        bad_hash["manifest_sha256"] = "0" * 64
        bad_hash["matrix_sha256"] = matrix_sha256(bad_hash)
        _expect_invalid(lambda: audit_calibration(manifest, plan, raw_path, bad_hash), "manifest sha")

        wrong_partition = copy.deepcopy(matrix)
        wrong_partition["partition_mode"] = "iid"
        wrong_partition["matrix_sha256"] = matrix_sha256(wrong_partition)
        _expect_invalid(
            lambda: audit_calibration(manifest, plan, raw_path, wrong_partition),
            "partition mode",
        )

        leaked = copy.deepcopy(manifest)
        leaked["protocol"]["calibration_seeds"][0] = 0
        leaked["calibration"]["seeds"][0] = 0
        _expect_invalid(lambda: audit_manifest(leaked), "seed")

        missing_path = root / "missing.csv"
        _write_rows(missing_path, rows[:-1])
        _expect_invalid(lambda: audit_calibration(manifest, plan, missing_path, matrix),
                        f"{manifest['calibration']['grid_size']} exact jobs")

        duplicate_path = root / "duplicate.csv"
        _write_rows(duplicate_path, rows + [rows[0]])
        _expect_invalid(lambda: audit_calibration(manifest, plan, duplicate_path, matrix),
                        "duplicate")

        missing_mechanism = [{key: value for key, value in row.items()
                              if key != "mechanism_signal"} for row in rows]
        missing_mechanism_path = root / "missing_mechanism.csv"
        _write_rows(missing_mechanism_path, missing_mechanism)
        _expect_invalid(lambda: audit_calibration(
            manifest, plan, missing_mechanism_path, matrix), "missing required columns")

        feasible_index = next(index for index, row in enumerate(rows)
                              if row["requested_action"] in
                              manifest["causes"][row["cause"]]["feasible_actions"])
        inactive = copy.deepcopy(rows)
        inactive[feasible_index]["activation_observed"] = "false"
        inactive[feasible_index]["mechanism_observed"] = "false"
        inactive[feasible_index]["effective_tool"] = "none"
        inactive_path = root / "inactive.csv"
        _write_rows(inactive_path, inactive)
        _expect_invalid(lambda: audit_calibration(manifest, plan, inactive_path, matrix),
                        "feasible action was not")

        bad_signal = copy.deepcopy(rows)
        bad_signal[feasible_index]["mechanism_signal"] = "tampered"
        bad_signal_path = root / "bad_signal.csv"
        _write_rows(bad_signal_path, bad_signal)
        _expect_invalid(lambda: audit_calibration(manifest, plan, bad_signal_path, matrix),
                        "mechanism signal")

        bad_noop = copy.deepcopy(rows)
        noop_index = next(index for index, row in enumerate(rows)
                          if row["requested_action"] == "no_op")
        bad_noop[noop_index]["effective_tool"] = "robust"
        bad_noop_path = root / "bad_noop.csv"
        _write_rows(bad_noop_path, bad_noop)
        _expect_invalid(lambda: audit_calibration(manifest, plan, bad_noop_path, matrix),
                        "effective tool")

        wrong_tool = copy.deepcopy(rows)
        wrong_tool[feasible_index]["effective_tool"] = "robust"
        wrong_tool_path = root / "wrong_tool.csv"
        _write_rows(wrong_tool_path, wrong_tool)
        _expect_invalid(lambda: audit_calibration(manifest, plan, wrong_tool_path, matrix),
                        "effective tool")

    print(
        "B3_ARTIFACT_AUDIT_OK"
        f" jobs={result['jobs']} manifest_sha256={result['manifest_sha256']}"
        " negatives=hash,partition,seed,missing,duplicate,mechanism,inactive,signal,noop,effective"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--code-sha")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check or not any((args.plan, args.raw, args.matrix)):
        self_check(args.manifest)
        return 0
    if not all((args.plan, args.raw, args.matrix, args.code_sha)):
        parser.error("--plan, --raw, --matrix, and --code-sha must be provided together")
    try:
        manifest = load_manifest(args.manifest)
        require_formal_manifest(manifest)
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        result = audit_calibration(
            manifest, plan, args.raw, matrix, args.manifest.parent, args.code_sha
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"B3_ARTIFACT_AUDIT_FAILED {exc}")
        return 1
    print(
        "B3_ARTIFACT_AUDIT_OK"
        f" jobs={result['jobs']} manifest_sha256={result['manifest_sha256']}"
        f" code_sha={result['code_sha']} matrix_sha256={result['matrix_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
