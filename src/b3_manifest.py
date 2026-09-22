"""Validate the frozen B3 suite manifest and its public/hidden boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from action_contract import ACTION_SPECS, B3_VISIBLE_ACTIONS
from evidence import ACTION_MECHANISM_SIGNAL, EVIDENCE_STATUSES, action_evidence_status
from observation_contract import PUBLIC_OBSERVATION_SCHEMA
from config import PARTITION_MODES, validate_partition_mode


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
DEFAULT_MANIFEST = ROOT / "b3_suite_manifest.json"
CURRENT_SCHEMA_VERSION = 2
CURRENT_BENCHMARK_VERSION = "b3-composite-v2"
CURRENT_STATUS = "refactor-open"
CURRENT_RESULT_ROOT = "results/b3_composite_v2"

VISIBLE_ACTIONS = set(B3_VISIBLE_ACTIONS)

FORMAL_CASES = {
    "real_dropout",
    "real_fault",
    "virtual_fault",
    "virtual_dropout",
    "label_dropout",
    "hetero_abrupt_dropout",
    "fault_dropout",
    "real_hetero",
    "virtual_hetero",
    "staggered",
}

HIDDEN_FIELDS = {
    "active_causes",
    "composition_seed",
    "affected_clients",
    "cause_intervals",
    "canonical_bundle",
    "feasible_action_sets",
    "event_schedule",
}


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_formal_manifest(manifest: dict) -> None:
    if manifest.get("status") != "frozen":
        raise ValueError(
            f"formal runs are disabled while manifest status={manifest.get('status')!r}"
        )


def require_versioned_output(path: Path, manifest: dict, partition_mode: str | None = None) -> None:
    expected = (REPOSITORY_ROOT / manifest["result_root"]).resolve()
    if partition_mode is not None:
        expected /= validate_partition_mode(partition_mode)
    actual = Path(path).resolve()
    try:
        actual.relative_to(expected)
    except ValueError as exc:
        raise ValueError(
            f"version/fingerprint mismatch: formal artifact must stay under {expected}"
        ) from exc


def partition_artifact_path(path: Path, manifest: dict, partition_mode: str) -> Path:
    """Insert the mode directly below the versioned B3 root when applicable."""
    mode = validate_partition_mode(partition_mode)
    path = Path(path)
    root = Path(manifest["result_root"])
    try:
        relative = path.resolve().relative_to((REPOSITORY_ROOT / root).resolve())
    except ValueError:
        return path if path.name == mode else path / mode
    if relative.parts and relative.parts[0] in PARTITION_MODES:
        if relative.parts[0] != mode:
            raise ValueError("artifact path partition mode mismatch")
        return path
    return root / mode / relative


def response_window(manifest: dict, event_round: int) -> tuple[int, int]:
    length = manifest.get("protocol", {}).get("response_window_length")
    if type(length) is not int or length <= 0:
        raise ValueError("response_window_length must be a positive integer")
    start = int(event_round)
    return start, start + length - 1


def canonical_bytes(manifest: dict) -> bytes:
    payload = copy.deepcopy(manifest)
    payload.get("integrity", {}).pop("manifest_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def public_contract(manifest: dict) -> dict:
    """Return only the contract that an Agent may receive during evaluation."""
    return {
        "benchmark_version": manifest["benchmark_version"],
        "episode_id_semantics": "opaque_identifier",
        "action_space": [
            name for name, row in manifest["actions"].items() if row["agent_visible"]
        ],
        "max_actions_per_bundle": 3,
        "observation_schema": manifest["public_observation_schema"],
    }


def validate_manifest(manifest: dict, root: Path = ROOT) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    protocol = manifest.get("protocol", {})
    actions = manifest.get("actions", {})
    causes = manifest.get("causes", {})
    cases = manifest.get("formal_cases", [])
    calibration = manifest.get("calibration", {})

    require(manifest.get("schema_version") == CURRENT_SCHEMA_VERSION,
            f"manifest schema must be {CURRENT_SCHEMA_VERSION}")
    require(manifest.get("benchmark_version") == CURRENT_BENCHMARK_VERSION,
            f"benchmark version must be {CURRENT_BENCHMARK_VERSION}")
    require(manifest.get("status") in {CURRENT_STATUS, "frozen"},
            "manifest status is invalid")
    require(manifest.get("result_root") == CURRENT_RESULT_ROOT,
            f"result root must be {CURRENT_RESULT_ROOT}")
    visible = {name for name, row in actions.items() if row.get("agent_visible")}
    require(visible == VISIBLE_ACTIONS, "real-Agent action space is not the approved 9 actions")
    require(set(actions) == VISIBLE_ACTIONS, "formal actions contain hidden or legacy actions")
    require(set(ACTION_MECHANISM_SIGNAL) == VISIBLE_ACTIONS,
            "mechanism-signal registry drifted from B3 actions")

    evaluation_seeds = set(protocol.get("evaluation_seeds", []))
    calibration_seeds = set(protocol.get("calibration_seeds", []))
    require(evaluation_seeds == {0, 44, 56}, "evaluation seeds changed")
    require(calibration_seeds == {101, 202, 303}, "calibration seeds changed")
    require(tuple(protocol.get("partition_modes", ())) == PARTITION_MODES,
            "partition modes changed")
    require(protocol.get("default_partition_mode") == "noniid",
            "default partition mode changed")
    phase6 = protocol.get("phase6", {})
    require(phase6.get("stress_seed") == 20260727, "Phase 6 stress seed changed")
    require(phase6.get("segment_rounds") == 10, "Phase 6 segment length changed")
    require(phase6.get("causes") == ["real_drift", "fault", "dropout"],
            "Phase 6 cause roster changed")
    require(phase6.get("stable_between_events") is True,
            "Phase 6 stable segments are required")
    require(phase6.get("agent_policy") == {
        "decision_cadence": 1, "timeout_seconds": 120.0,
        "retry_delays_seconds": [2, 5, 10, 20, 40], "max_calls": 70,
    }, "Phase 6 Agent policy changed")
    require(evaluation_seeds.isdisjoint(calibration_seeds), "calibration/evaluation seed leak")
    require(protocol.get("case_selection") == "all_allowed_cases", "formal cases are sampled")
    require(type(protocol.get("response_window_length")) is int and
            protocol["response_window_length"] > 0,
            "response_window_length must be positive")
    require("response_window" not in protocol, "absolute response_window is forbidden")

    case_ids = [case.get("case_id") for case in cases]
    require(len(case_ids) == len(set(case_ids)), "duplicate formal case_id")
    require(set(case_ids) == FORMAL_CASES, "formal case pool is not the approved ten cases")
    require(set(protocol.get("formal_case_ids", [])) == FORMAL_CASES, "protocol case list mismatch")
    require(protocol.get("causes_per_episode") == [1, 2],
            "formal cases must contain one or two causes")
    require(protocol.get("first_smoke_case_id") == "real_dropout", "first smoke case changed")

    stages = set(manifest.get("execution_order", [])) | {"control"}
    for name, action in actions.items():
        require(action.get("family"), f"{name}: missing family")
        require(action.get("evidence") in EVIDENCE_STATUSES,
                f"{name}: invalid evidence")
        require(action.get("evidence") == action_evidence_status(name),
                f"{name}: evidence drifted from registry")
        require(action.get("execution_stage") in stages, f"{name}: invalid execution stage")
        require(action.get("lifecycle") in {
            "control", "reversible_state", "per_round", "one_shot_irreversible"
        }, f"{name}: invalid lifecycle")
        spec = ACTION_SPECS[name]
        require(action.get("family") == spec.family, f"{name}: family drifted from runtime")
        require(action.get("lifecycle") == spec.lifecycle,
                f"{name}: lifecycle drifted from runtime")

    for cause_id, cause in causes.items():
        feasible = cause.get("feasible_actions", [])
        require(len(feasible) == len(set(feasible)), f"{cause_id}: duplicate feasible action")
        require(set(feasible) <= set(actions), f"{cause_id}: unknown feasible action")
        canonical = cause.get("canonical_action")
        require(canonical is None or canonical in feasible, f"{cause_id}: illegal canonical action")
        if cause.get("status") == "supported":
            require(bool(feasible), f"{cause_id}: supported cause has no feasible action")
    require(causes.get("recurrent_drift", {}).get("feasible_actions") == [],
            "recurrent hard case gained an unfrozen feasible action")
    require(causes.get("stable", {}).get("feasible_actions") == ["no_op"],
            "stable must map only to no_op")

    forbidden_pairs = [set(row.get("causes", [])) for row in manifest.get("forbidden_combinations", [])]
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        active_causes = case.get("causes", [])
        require(len(active_causes) in protocol["causes_per_episode"]
                and len(active_causes) == len(set(active_causes)),
                f"{case_id}: formal case must contain distinct approved causes")
        require((case_id == "staggered") == (active_causes == ["staggered_concepts"]),
                f"{case_id}: only staggered may be a single-cause formal case")
        require(set(active_causes) <= set(causes), f"{case_id}: unknown cause")
        require("recurrent_drift" not in active_causes, f"{case_id}: recurrent is forbidden")
        require(not any("*" not in pair and pair == set(active_causes) for pair in forbidden_pairs),
                f"{case_id}: explicitly forbidden combination")

        feasible_bundles = case.get("feasible_bundles", [])
        canonical_bundle = case.get("canonical_bundle", [])
        require(canonical_bundle in feasible_bundles, f"{case_id}: canonical bundle not feasible")
        for bundle in feasible_bundles:
            require(1 <= len(bundle) <= 3, f"{case_id}: bundle size outside 1..3")
            require(len(bundle) == len(set(bundle)), f"{case_id}: duplicate bundle action")
            require(set(bundle) <= visible, f"{case_id}: bundle contains non-visible action")
            require(all(actions[action]["evidence"] in
                        {"algorithm_supported", "source_aligned"} for action in bundle),
                    f"{case_id}: formal bundle contains proxy/unsupported action")
            require(not ("no_op" in bundle and len(bundle) > 1), f"{case_id}: no_op conflict")
            families = [actions[action]["family"] for action in bundle if action in actions]
            require(len(families) == len(set(families)), f"{case_id}: same-family conflict")
            for cause_id in active_causes:
                feasible = set(causes.get(cause_id, {}).get("feasible_actions", []))
                require(bool(feasible.intersection(bundle)), f"{case_id}: bundle misses {cause_id}")
            covered = set().union(*(
                set(causes.get(cause_id, {}).get("feasible_actions", []))
                for cause_id in active_causes
            ))
            require(set(bundle) <= covered, f"{case_id}: feasible bundle has an extra action")

    public_fields = set(manifest.get("public_observation_schema", {}))
    hidden_fields = set(manifest.get("hidden_event_schema", {}))
    require(hidden_fields == HIDDEN_FIELDS, "hidden event schema changed")
    require(public_fields.isdisjoint(hidden_fields), "hidden fields leak into public observations")
    require(manifest.get("public_observation_schema") == PUBLIC_OBSERVATION_SCHEMA,
            "public observation schema drifted from runtime")
    require("scenario" not in public_fields and "event_round" not in public_fields,
            "ground-truth scenario/event field is public")
    public_blob = json.dumps(public_contract(manifest), sort_keys=True)
    require(not any(f'"{field}"' in public_blob for field in HIDDEN_FIELDS),
            "public contract contains scorer-only fields")

    require(set(calibration.get("seeds", [])) == calibration_seeds,
            "calibration block seed mismatch")
    require(set(calibration.get("actions", [])) == visible,
            "calibration grid does not cover all visible actions")
    expected_grid = (
        len(calibration.get("seeds", []))
        * len(calibration.get("causes", []))
        * len(calibration.get("actions", []))
    )
    require(calibration.get("grid_size") == expected_grid == 189,
            "calibration grid must contain 189 runs")

    baseline = manifest.get("baseline", {})
    freeze_path = root / baseline.get("freeze_path", "")
    require(freeze_path.is_file(), "baseline freeze file is missing")
    if freeze_path.is_file():
        freeze_bytes = freeze_path.read_bytes()
        freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
        freeze = json.loads(freeze_bytes)
        expected_freeze_sha = baseline.get("freeze_sha256")
        require(
            freeze_sha == expected_freeze_sha
            or freeze.get("redacted_from_sha256") == expected_freeze_sha,
            "baseline freeze SHA mismatch",
        )
        require(freeze.get("freeze_id") == baseline.get("freeze_id"), "baseline freeze_id mismatch")
        require(freeze.get("integrity", {}).get("git_sha") == baseline.get("code_sha"),
                "baseline code SHA mismatch")

    computed = manifest_sha256(manifest)
    require(manifest.get("integrity", {}).get("manifest_sha256") == computed,
            f"manifest SHA mismatch (computed {computed})")
    require(canonical_bytes(manifest) == canonical_bytes(copy.deepcopy(manifest)),
            "canonical serialization is not deterministic")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--public", action="store_true", help="print the Agent-visible contract")
    parser.add_argument("--canonical", action="store_true", help="print canonical hash input")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        print("B3_MANIFEST_INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    if args.public:
        print(json.dumps(public_contract(manifest), indent=2, ensure_ascii=False))
    elif args.canonical:
        print(canonical_bytes(manifest).decode("utf-8"))
    else:
        print(
            "B3_MANIFEST_OK"
            f" sha256={manifest_sha256(manifest)}"
            f" cases={len(manifest['formal_cases'])}"
            f" actions={len(manifest['actions'])}"
            f" calibration_runs={manifest['calibration']['grid_size']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
