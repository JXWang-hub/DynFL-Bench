"""Scorer-only B3 episode plans with opaque persisted episode identifiers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import secrets
import tempfile
from pathlib import Path
from config import validate_partition_mode

from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256, validate_manifest
from episode_timeline import compile_fdms_dropout_intervals


ROOT = Path(__file__).resolve().parent
DATA_CAUSES = {"real_drift", "virtual_drift", "label_prior_drift"}
ATOMIC_EXTENSION_CASES = {
    "atomic_real": ("real_drift", "drift_adapt"),
    "atomic_virtual": ("virtual_drift", "moment_align_adapt"),
    "atomic_label": ("label_prior_drift", "label_prior_adapt"),
    "atomic_fault": ("fault", "robust"),
    "atomic_dropout": ("dropout", "dropout_handle"),
    "atomic_hetero": ("partial_participation_hetero", "select_clients"),
}


def canonical_plan_bytes(plan: dict) -> bytes:
    payload = copy.deepcopy(plan)
    payload.pop("plan_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def plan_sha256(plan: dict) -> str:
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def _token(composition_seed: int, case_id: str, cause: str) -> str:
    raw = f"{composition_seed}:{case_id}:{cause}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _ranked_clients(composition_seed: int, case_id: str, cause: str,
                    num_clients: int) -> list[int]:
    prefix = f"{composition_seed}:{case_id}:{cause}:"
    return sorted(
        range(num_clients),
        key=lambda client: hashlib.sha256(f"{prefix}{client}".encode("utf-8")).digest(),
    )


def _count(rate: float, num_clients: int, leave_one: bool = False) -> int:
    count = max(1, int(float(rate) * num_clients + 0.5))
    return min(num_clients - 1 if leave_one else num_clients, count)


def _load_inputs(manifest_path: Path) -> tuple[dict, dict]:
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    if errors and manifest.get("atomic_extension"):
        errors = _atomic_extension_errors(manifest)
    if errors:
        raise ValueError("invalid B3 manifest: " + "; ".join(errors))
    freeze_path = ROOT / manifest["baseline"]["freeze_path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    return manifest, freeze["protocol"]["scenario_workpoints"]


def _atomic_extension_errors(manifest: dict) -> list[str]:
    """Accept only the narrow, provenance-linked single-cause extension."""
    base = load_manifest(DEFAULT_MANIFEST)
    extension = manifest.get("atomic_extension", {})
    expected_cases = [
        {"case_id": case_id, "causes": [cause], "canonical_bundle": [action],
         "feasible_bundles": [[action]]}
        for case_id, (cause, action) in ATOMIC_EXTENSION_CASES.items()
    ]
    errors = []
    if extension != {"base_manifest_sha256": manifest_sha256(base),
                     "cases": [case_id.removeprefix("atomic_")
                               for case_id in ATOMIC_EXTENSION_CASES]}:
        errors.append("atomic extension provenance mismatch")
    if manifest.get("formal_cases", [])[len(base["formal_cases"]):] != expected_cases:
        errors.append("atomic extension case roster mismatch")
    candidate = copy.deepcopy(manifest)
    candidate.pop("atomic_extension", None)
    candidate["formal_cases"] = candidate.get("formal_cases", [])[:len(base["formal_cases"])]
    candidate.setdefault("integrity", {})["manifest_sha256"] = (
        base.get("integrity", {}).get("manifest_sha256")
    )
    if candidate != base:
        errors.append("atomic extension changed frozen manifest content")
    if manifest.get("integrity", {}).get("manifest_sha256") != manifest_sha256(manifest):
        errors.append("atomic extension manifest SHA mismatch")
    return errors


def _affected_clients(cause: str, case_id: str, num_clients: int,
                      composition_seed: int, workpoint: dict, event_round: int) -> dict:
    if cause in DATA_CAUSES:
        return {"resolver": "all_clients", "clients": list(range(num_clients))}
    if cause == "staggered_concepts":
        return {"resolver": "all_clients", "clients": list(range(num_clients))}
    if cause == "fault":
        count = _count(workpoint["byzantine_frac"], num_clients, leave_one=True)
        return {
            "resolver": "fixed_sha256_rank",
            "clients": _ranked_clients(composition_seed, case_id, cause, num_clients)[:count],
            "fraction": workpoint["byzantine_frac"],
            "resolver_token": _token(composition_seed, case_id, cause),
        }
    if cause == "dropout":
        if workpoint["dropout_mode"] == "fdms_clustered":
            return {
                "resolver": "fdms_clustered_roster",
                "clients": None,
                "mode": workpoint["dropout_mode"],
                "target_rate": workpoint["dropout_rate"],
                "start_round": event_round,
                "recover_round": workpoint["dropout_recover_round"],
                "warmup_rounds": workpoint.get("fdms_warmup_rounds", 1),
            }
        count = _count(workpoint["dropout_rate"], num_clients, leave_one=True)
        return {
            "resolver": "dominant_label_lower_half",
            "clients": None,
            "mode": workpoint["dropout_mode"],
            "target_rate": workpoint["dropout_rate"],
            "target_count": count,
            "start_round": event_round,
            "recover_round": workpoint["dropout_recover_round"],
            "tie_break_token": _token(composition_seed, case_id, cause),
        }
    if cause == "partial_participation_hetero":
        rate = workpoint["participation"]
        return {
            "resolver": "scheduler_planned_subset",
            "clients": None,
            "planned_rate": rate,
            "planned_count": _count(rate, num_clients),
            "resolver_token": _token(composition_seed, case_id, cause),
        }
    raise ValueError(f"Phase 3A has no target resolver for {cause}")


def build_episode_plan(case_id: str, training_seed: int, num_clients: int = 20,
                       manifest_path: Path = DEFAULT_MANIFEST,
                       episode_id: str | None = None,
                       partition_mode: str = "noniid") -> dict:
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    manifest, workpoints = _load_inputs(manifest_path)
    cases = {case["case_id"]: case for case in manifest["formal_cases"]}
    if case_id not in cases:
        raise ValueError(f"unknown formal case_id: {case_id}")

    case = cases[case_id]
    partition_mode = validate_partition_mode(partition_mode)
    protocol = manifest["protocol"]
    event_round = int(protocol["event_round"])
    activation_round = 0 if case_id == "atomic_hetero" else event_round
    composition_seed = int(protocol["composition_seed"])
    suite_sha = manifest_sha256(manifest)
    episode_id = episode_id or "ep_" + secrets.token_hex(16)
    if not re.fullmatch(r"ep_[0-9a-f]{32}", episode_id):
        raise ValueError("episode_id must be an opaque 128-bit lowercase hex nonce")
    affected = {
        cause: _affected_clients(
            cause, case_id, num_clients, composition_seed, workpoints[cause], activation_round
        )
        for cause in case["causes"]
    }
    hidden_event = {
        "active_causes": case["causes"],
        "composition_seed": composition_seed,
        "affected_clients": affected,
        "canonical_bundle": case["canonical_bundle"],
        "feasible_action_sets": {
            cause: manifest["causes"][cause]["feasible_actions"]
            for cause in case["causes"]
        },
        "event_schedule": [{
            "round": activation_round,
            "activate": case["causes"],
            "severity": {cause: workpoints[cause] for cause in case["causes"]},
        }],
    }
    plan = {
        "schema_version": manifest["schema_version"],
        "benchmark_version": manifest["benchmark_version"],
        "suite_manifest_sha256": suite_sha,
        "episode_id": episode_id,
        "case_id": case_id,
        "partition_mode": partition_mode,
        "workpoint_role": ("iid_system_control" if partition_mode == "iid" and
                           case_id == "hetero_abrupt_dropout" else "benchmark"),
        "training_seed": int(training_seed),
        "num_clients": int(num_clients),
        "total_rounds": int(protocol["rounds"]),
        "hidden_event": hidden_event,
    }
    plan["plan_sha256"] = plan_sha256(plan)
    return plan


def validate_episode_plan(plan: dict, case_id: str, training_seed: int,
                          num_clients: int, manifest_path: Path = DEFAULT_MANIFEST,
                          partition_mode: str = "noniid") -> None:
    manifest = load_manifest(manifest_path)
    expected = {
        "schema_version": manifest["schema_version"],
        "benchmark_version": manifest["benchmark_version"],
        "suite_manifest_sha256": manifest_sha256(manifest),
        "case_id": case_id,
        "partition_mode": validate_partition_mode(partition_mode),
        "workpoint_role": ("iid_system_control" if partition_mode == "iid" and
                           case_id == "hetero_abrupt_dropout" else "benchmark"),
        "training_seed": int(training_seed),
        "num_clients": int(num_clients),
        "total_rounds": int(manifest["protocol"]["rounds"]),
    }
    if plan.get("plan_sha256") != plan_sha256(plan):
        raise ValueError("episode plan SHA mismatch")
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("episode plan version/fingerprint mismatch")
    if not re.fullmatch(r"ep_[0-9a-f]{32}", str(plan.get("episode_id", ""))):
        raise ValueError("episode plan has an invalid opaque episode_id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(plan.get("partition_sha256", ""))):
        raise ValueError("episode plan version/fingerprint mismatch: partition")
    hidden = plan.get("hidden_event", {})
    intervals = hidden.get("cause_intervals")
    if not isinstance(intervals, dict) or set(intervals) != set(hidden.get("active_causes", ())):
        raise ValueError("episode plan version/fingerprint mismatch: cause_intervals")
    for cause, by_client in intervals.items():
        if not isinstance(by_client, dict):
            raise ValueError(f"invalid cause intervals for {cause}")
        for raw_client, spans in by_client.items():
            if not str(raw_client).isdigit() or not 0 <= int(raw_client) < num_clients:
                raise ValueError(f"invalid interval client for {cause}")
            previous_end = -1
            for span in spans:
                if (not isinstance(span, list) or len(span) != 2 or
                        type(span[0]) is not int or
                        (span[1] is not None and type(span[1]) is not int)):
                    raise ValueError(f"invalid interval for {cause}/{raw_client}")
                start, end = span
                if start < previous_end or (end is not None and end <= start):
                    raise ValueError(f"overlapping or empty interval for {cause}/{raw_client}")
                previous_end = end if end is not None else 10**18


def load_private_plan(path: Path, case_id: str, training_seed: int,
                      num_clients: int, manifest_path: Path = DEFAULT_MANIFEST,
                      partition_mode: str = "noniid") -> dict:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load private episode plan: {path}") from exc
    validate_episode_plan(
        plan, case_id, training_seed, num_clients, manifest_path, partition_mode
    )
    return plan


def save_private_plan(path: Path, plan: dict) -> None:
    path = Path(path)
    if plan.get("plan_sha256") != plan_sha256(plan):
        raise ValueError("refusing to save an invalid episode plan")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_plan_bytes(existing) != canonical_plan_bytes(plan):
            raise ValueError(f"refusing to overwrite a different private episode plan: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def bind_partition(plan: dict, partition_sha256: str, partition_mode: str | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", str(partition_sha256)):
        raise ValueError("partition_sha256 must be a lowercase SHA-256")
    bound = copy.deepcopy(plan)
    if partition_mode is not None and bound.get("partition_mode") != validate_partition_mode(partition_mode):
        raise ValueError("episode plan version/fingerprint mismatch: partition mode")
    existing = bound.get("partition_sha256")
    if existing is not None and existing != partition_sha256:
        raise ValueError("episode plan version/fingerprint mismatch: partition")
    bound["partition_sha256"] = partition_sha256
    bound["plan_sha256"] = plan_sha256(bound)
    return bound


def _client_ids(values, num_clients: int, label: str) -> list[int]:
    try:
        clients = list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of client ids") from exc
    if any(type(client) is not int for client in clients):
        raise ValueError(f"{label} must contain integer client ids")
    if len(clients) != len(set(clients)):
        raise ValueError(f"{label} contains duplicate client ids")
    if any(client < 0 or client >= num_clients for client in clients):
        raise ValueError(f"{label} contains an out-of-range client id")
    return clients


def resolve_episode_clients(plan: dict, dominant_labels: dict | None = None,
                            planned_clients=None, n_classes: int | None = None,
                            fdms_groups=None) -> dict:
    """Resolve metadata-dependent client sets without training imports or mutation."""
    if plan.get("plan_sha256") != plan_sha256(plan):
        raise ValueError("episode plan SHA mismatch")
    num_clients = plan.get("num_clients")
    if type(num_clients) is not int or num_clients < 2:
        raise ValueError("episode plan has invalid num_clients")
    try:
        hidden = plan["hidden_event"]
        causes = hidden["active_causes"]
        targets = hidden["affected_clients"]
    except (KeyError, TypeError) as exc:
        raise ValueError("episode plan is missing hidden client targets") from exc
    if len(causes) != len(set(causes)) or set(causes) != set(targets):
        raise ValueError("active causes and client targets do not match")

    resolved = copy.deepcopy(plan)
    resolved_targets = resolved["hidden_event"]["affected_clients"]
    existing_intervals = resolved["hidden_event"].get("cause_intervals", {})
    dynamic_intervals = {}
    for cause in causes:
        target = resolved_targets[cause]
        resolver = target.get("resolver")
        if resolver not in {
            "all_clients", "fixed_sha256_rank", "dominant_label_lower_half",
            "scheduler_planned_subset", "fdms_clustered_roster",
        }:
            raise ValueError(f"unknown client resolver for {cause}: {resolver}")
        if resolver == "fdms_clustered_roster":
            if cause in existing_intervals:
                dynamic_intervals[cause] = existing_intervals[cause]
                target["clients"] = sorted(map(int, existing_intervals[cause]))
            else:
                spans = compile_fdms_dropout_intervals(
                    num_clients, fdms_groups or (), target["target_rate"],
                    plan["training_seed"], target["start_round"], plan["total_rounds"],
                    target.get("warmup_rounds", 0), target.get("recover_round", 0),
                )
                dynamic_intervals[cause] = {
                    str(client): [list(span) for span in intervals]
                    for client, intervals in spans.items()
                }
                target["clients"] = sorted(spans)
            continue
        if target.get("clients") is not None:
            clients = _client_ids(target["clients"], num_clients, f"{cause} clients")
            expected = (num_clients if resolver == "all_clients" else
                        target.get("target_count") if resolver == "dominant_label_lower_half" else
                        target.get("planned_count") if resolver == "scheduler_planned_subset" else None)
            if expected is not None and len(clients) != expected:
                raise ValueError(f"{cause} client count does not match its plan")
            continue

        if resolver == "dominant_label_lower_half":
            if not isinstance(dominant_labels, dict):
                raise ValueError("dominant_labels must be a client-to-label mapping")
            label_clients = _client_ids(dominant_labels, num_clients, "dominant_labels keys")
            if set(label_clients) != set(range(num_clients)):
                raise ValueError("dominant_labels must cover every client exactly once")
            if type(n_classes) is not int or n_classes < 2:
                raise ValueError("n_classes must be provided for dropout resolution")
            if any(type(label) is not int or not 0 <= label < n_classes
                   for label in dominant_labels.values()):
                raise ValueError("dominant_labels contains an invalid class id")
            count = target.get("target_count")
            if type(count) is not int or not 0 < count < num_clients:
                raise ValueError("dropout target_count is invalid")
            token = target.get("tie_break_token")
            if not isinstance(token, str) or not token:
                raise ValueError("dropout tie_break_token is missing")
            half = n_classes // 2
            ranked = sorted(
                range(num_clients),
                key=lambda client: (
                    dominant_labels[client] >= half,
                    hashlib.sha256(f"{token}:{client}".encode("utf-8")).digest(),
                ),
            )
            target["clients"] = sorted(ranked[:count])
        elif resolver == "scheduler_planned_subset":
            clients = _client_ids(planned_clients, num_clients, "planned_clients")
            if len(clients) != target.get("planned_count"):
                raise ValueError("planned_clients count does not match the scheduler contract")
            target["clients"] = sorted(clients)
        else:
            raise ValueError(f"{cause} resolver unexpectedly has no clients")

    if "fault" in causes and "dropout" in causes:
        dropped = set(resolved_targets["dropout"]["clients"])
        online = set(range(num_clients)) - dropped
        fault = resolved_targets["fault"]
        count = _count(fault["fraction"], len(online), leave_one=True)
        ranked = _ranked_clients(
            hidden["composition_seed"], plan["case_id"], "fault", num_clients
        )
        fault["clients"] = [client for client in ranked if client in online][:count]
        fault["population"] = "event_online_roster"

    schedule = resolved["hidden_event"]["event_schedule"]
    resolved["hidden_event"]["cause_intervals"] = {}
    for cause in causes:
        target = resolved_targets[cause]
        if target.get("resolver") == "fdms_clustered_roster":
            resolved["hidden_event"]["cause_intervals"][cause] = dynamic_intervals[cause]
            continue
        start = (0 if cause == "partial_participation_hetero" else
                 min(int(item["round"]) for item in schedule if cause in item["activate"]))
        end = (target.get("recover_round") or None) if cause == "dropout" else None
        resolved["hidden_event"]["cause_intervals"][cause] = {
            str(client): [[start, end]] for client in target["clients"]
        }

    resolved["plan_sha256"] = plan_sha256(resolved)
    return resolved


def public_contract(plan: dict) -> dict:
    return {"episode_id": plan["episode_id"]}


def self_check(manifest_path: Path = DEFAULT_MANIFEST) -> None:
    if not __debug__:
        raise RuntimeError("episode-plan self-check refuses optimized mode")
    manifest, _ = _load_inputs(manifest_path)
    expected_ids = manifest["protocol"]["formal_case_ids"]
    cases = {case["case_id"]: case for case in manifest["formal_cases"]}
    episode_ids = set()
    dominant_labels = {client: client % 10 for client in range(20)}
    planned_clients = [19, 2, 17, 4, 15, 6, 13, 8]
    for case_id in expected_ids:
        reference_targets = None
        reference_resolved_targets = None
        for seed in manifest["protocol"]["evaluation_seeds"]:
            plan = build_episode_plan(case_id, seed, manifest_path=manifest_path)
            repeat = build_episode_plan(
                case_id, seed, manifest_path=manifest_path,
                episode_id=plan["episode_id"],
            )
            fresh = build_episode_plan(case_id, seed, manifest_path=manifest_path)
            assert canonical_plan_bytes(plan) == canonical_plan_bytes(repeat)
            assert fresh["episode_id"] != plan["episode_id"]
            assert fresh["hidden_event"] == plan["hidden_event"]
            assert plan["plan_sha256"] == plan_sha256(plan)
            assert re.fullmatch(r"ep_[0-9a-f]{32}", plan["episode_id"])
            assert plan["episode_id"] not in episode_ids
            episode_ids.add(plan["episode_id"])

            hidden = plan["hidden_event"]
            case = cases[case_id]
            assert hidden["active_causes"] == case["causes"]
            assert hidden["canonical_bundle"] == case["canonical_bundle"]
            expected_round = 0 if case_id == "atomic_hetero" else manifest["protocol"]["event_round"]
            assert hidden["event_schedule"][0]["round"] == expected_round
            assert set(hidden["affected_clients"]) == set(case["causes"])
            assert public_contract(plan) == {"episode_id": plan["episode_id"]}
            assert not any(
                token in json.dumps(public_contract(plan), sort_keys=True)
                for token in (case_id, "active_causes", "event_schedule", "canonical_bundle")
            )

            targets = hidden["affected_clients"]
            if reference_targets is None:
                reference_targets = targets
            else:
                assert targets == reference_targets, "composition changed with training seed"
            for cause in case["causes"]:
                target = targets[cause]
                if cause in DATA_CAUSES:
                    assert target["clients"] == list(range(plan["num_clients"]))
                elif cause == "staggered_concepts":
                    assert target["clients"] == list(range(plan["num_clients"]))
                elif cause == "fault":
                    expected = _count(target["fraction"], plan["num_clients"], leave_one=True)
                    assert len(target["clients"]) == expected == len(set(target["clients"]))
                elif cause == "dropout":
                    assert target["clients"] is None
                    assert target["resolver"] == "dominant_label_lower_half"
                    assert target["target_count"] == 10
                elif cause == "partial_participation_hetero":
                    assert target["clients"] is None and target["planned_count"] == 8

            resolved = resolve_episode_clients(
                plan, dominant_labels, planned_clients, n_classes=10
            )
            assert plan["plan_sha256"] == plan_sha256(plan), "resolver mutated its input"
            assert resolved["plan_sha256"] == plan_sha256(resolved)
            assert public_contract(resolved) == public_contract(plan)
            resolved_targets = resolved["hidden_event"]["affected_clients"]
            if reference_resolved_targets is None:
                reference_resolved_targets = resolved_targets
            else:
                assert resolved_targets == reference_resolved_targets
            for cause in case["causes"]:
                clients = resolved_targets[cause]["clients"]
                assert len(clients) == len(set(clients))
                if cause == "dropout":
                    assert len(clients) == 10
                    assert all(dominant_labels[client] < 5 for client in clients)
                elif cause == "fault" and "dropout" in case["causes"]:
                    dropped = set(resolved_targets["dropout"]["clients"])
                    assert len(clients) == 2 and not set(clients) & dropped
                elif cause == "partial_participation_hetero":
                    assert clients == sorted(planned_clients)

    assert set(cases) == set(expected_ids) and len(expected_ids) == 10
    print(
        "B3_PHASE3A_OK"
        f" cases={len(expected_ids)} episodes={len(episode_ids)}"
        f" manifest_sha256={manifest_sha256(manifest)}"
    )
    try:
        resolve_episode_clients(
            build_episode_plan("real_dropout", 0, manifest_path=manifest_path),
            {client: client % 10 for client in range(19)}, n_classes=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete dominant_labels did not fail closed")
    try:
        resolve_episode_clients(
            build_episode_plan("hetero_abrupt_dropout", 0, manifest_path=manifest_path),
            dominant_labels, [0] * 8, n_classes=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate planned_clients did not fail closed")
    sparse_lower_half = {client: 0 if client < 4 else 9 for client in range(20)}
    filled = resolve_episode_clients(
        build_episode_plan("real_dropout", 0, manifest_path=manifest_path),
        sparse_lower_half, n_classes=10,
    )["hidden_event"]["affected_clients"]["dropout"]["clients"]
    assert len(filled) == 10 and set(range(4)) <= set(filled)
    fdms_plan = build_episode_plan("real_dropout", 0, manifest_path=manifest_path)
    fdms_target = fdms_plan["hidden_event"]["affected_clients"]["dropout"]
    fdms_target.clear()
    fdms_target.update({
        "resolver": "fdms_clustered_roster", "clients": None,
        "mode": "fdms_clustered", "target_rate": 0.5,
        "start_round": manifest["protocol"]["event_round"],
        "recover_round": 0, "warmup_rounds": 1,
    })
    fdms_plan["plan_sha256"] = plan_sha256(fdms_plan)
    groups = [list(range(start, start + 4)) for start in range(0, 20, 4)]
    fdms_resolved = resolve_episode_clients(
        fdms_plan, dominant_labels, n_classes=10, fdms_groups=groups,
    )
    fdms_repeat = resolve_episode_clients(
        fdms_plan, dominant_labels, n_classes=10, fdms_groups=groups,
    )
    assert fdms_resolved["hidden_event"]["cause_intervals"]["dropout"] == (
        fdms_repeat["hidden_event"]["cause_intervals"]["dropout"]
    )
    fdms_timeline = fdms_resolved["hidden_event"]["cause_intervals"]["dropout"]
    for round_idx in range(manifest["protocol"]["event_round"], fdms_plan["total_rounds"]):
        absent = sum(any(start <= round_idx < end for start, end in spans)
                     for spans in fdms_timeline.values())
        assert absent == 10
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "private_plan.json"
        persisted = bind_partition(resolve_episode_clients(
            build_episode_plan("real_dropout", 0, manifest_path=manifest_path),
            dominant_labels, n_classes=10,
        ), "0" * 64)
        save_private_plan(path, persisted)
        loaded = load_private_plan(path, "real_dropout", 0, 20, manifest_path)
        assert canonical_plan_bytes(loaded) == canonical_plan_bytes(persisted)
        try:
            bind_partition(persisted, "1" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("partition mismatch did not fail closed")
        tampered = copy.deepcopy(persisted)
        tampered["training_seed"] = 1
        path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            load_private_plan(path, "real_dropout", 0, 20, manifest_path)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered private plan did not fail closed")
    print("B3_PHASE3AR_OK cases=10 dropout_clients=10 hetero_planned_clients=8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--case-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    if args.check or not args.case_id:
        self_check(args.manifest)
        return 0
    plan = build_episode_plan(args.case_id, args.seed, args.num_clients, args.manifest)
    print(json.dumps(public_contract(plan) if args.public else plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
