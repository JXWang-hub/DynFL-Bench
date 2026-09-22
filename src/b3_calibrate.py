"""Run the frozen B3 atomic-action calibration grid on single-cause scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from agents import Action, Agent, SelectClientsAgent
from action_contract import ACTION_SPECS
from b3_audit import FEDDRIFT_FIELDS, aggregate_cause_rows, audit_calibration
from b3_manifest import (
    DEFAULT_MANIFEST, load_manifest, manifest_sha256, validate_manifest,
    partition_artifact_path, require_formal_manifest, require_versioned_output,
)
from b3_fingerprint import source_sha256
from b3_scoring import feddrift_structure_summary
from engine import Engine
from evidence import ACTION_MECHANISM_SIGNAL
from multi_seed import make_cfg
from run import build_data, set_seed
from config import PARTITION_MODES


CAUSE_CONFIG = {
    "real_drift": ("drift", "real"),
    "virtual_drift": ("drift", "virtual"),
    "label_prior_drift": ("drift", "label"),
    "fault": ("fault", None),
    "dropout": ("dropout", None),
    "partial_participation_hetero": ("hetero", None),
    "staggered_concepts": ("staggered", None),
}

RAW_FIELDS = [
    "run_id", "manifest_sha256", "code_sha", "source_sha256", "git_dirty", "git_status_sha256",
    "dataset", "partition_mode", "smoke", "cause", "scenario", "drift_type", "seed",
    "requested_action", "effective_tool", "action_emitted", "activation_observed",
    "mechanism_observed", "mechanism_signal", "event_round", "rounds",
    "post_event_mean_accuracy", "post_event_worst_class_accuracy", "final_accuracy",
    "switch_round", "duration_seconds",
    *FEDDRIFT_FIELDS,
]


class FixedActionAgent(Agent):
    """Activate one tool at the event; repeat only the per-round selection tool."""

    def __init__(self, action: str, event_round: int):
        self.action = action
        self.event_round = event_round
        self.name = f"fixed_{action}"
        self.fired = False
        self.selector = SelectClientsAgent()

    def decide(self, obs):
        if int(obs["round"]) < self.event_round or self.action == "no_op":
            return Action("no_op")
        if self.action == "select_clients":
            forced = dict(obs)
            forced["select_enabled"] = True
            return self.selector.decide(forced)
        if ACTION_SPECS[self.action].lifecycle == "reversible_state":
            return Action(self.action)
        if not self.fired:
            self.fired = True
            return Action(self.action)
        return Action("no_op")


def git_metadata(root: Path) -> tuple[str, bool, str]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=True
        )
        return proc.stdout.strip()

    code_sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return code_sha, bool(status), hashlib.sha256(status.encode("utf-8")).hexdigest()


def selected(available: list, requested: list | None, label: str) -> list:
    if not requested:
        return list(available)
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(map(str, unknown)))}")
    return [item for item in available if item in requested]


def build_plan(manifest: dict, causes: list[str], actions: list[str], seeds: list[int],
               dataset: str, smoke: bool, partition_mode: str = "noniid",
               hetero_beta: float | None = None,
               dropout_beta: float | None = None,
               dropout_cache_ttl: int | None = None) -> dict:
    suite_sha = manifest_sha256(manifest)
    jobs = []
    for cause in causes:
        scenario, drift_type = CAUSE_CONFIG[cause]
        for seed in seeds:
            for action in actions:
                identity = (f"{suite_sha}:{dataset}:{partition_mode}:{int(smoke)}:"
                            f"{cause}:{seed}:{action}")
                if hetero_beta is not None:
                    identity += f":hetero_beta={hetero_beta:g}"
                if dropout_beta is not None:
                    identity += f":dropout_beta={dropout_beta:g}"
                if dropout_cache_ttl is not None:
                    identity += f":dropout_cache_ttl={dropout_cache_ttl}"
                jobs.append({
                    "run_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    "cause": cause,
                    "scenario": scenario,
                    "drift_type": drift_type,
                    "seed": seed,
                    "action": action,
                })
    plan = {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": suite_sha,
        "dataset": dataset,
        "partition_mode": partition_mode,
        "smoke": smoke,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    overrides = {}
    if hetero_beta is not None:
        overrides["hetero_beta"] = hetero_beta
    if dropout_beta is not None:
        overrides["dropout_beta"] = dropout_beta
    if dropout_cache_ttl is not None:
        overrides["dropout_cache_ttl"] = dropout_cache_ttl
    if overrides:
        plan["config_overrides"] = overrides
    return plan


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mechanism_result(action: str, hist: dict, event_round: int) -> tuple[bool, str]:
    rows = hist.get("telemetry", [])[event_round + 1:]
    checks = {
        "no_op": (hist.get("tool") == "none", "no_active_tool"),
        "drift_adapt": (any(row.get("using_drift_adapt") for row in rows), "using_drift_adapt"),
        "moment_align_adapt": (any(row.get("moment_align_enabled") for row in rows), "moment_align_enabled"),
        "label_prior_adapt": (any(row.get("label_prior_adapt_enabled") for row in rows), "label_prior_adapt_enabled"),
        "robust": (any(row.get("robust_variant") not in (None, "", "n/a", "none") for row in rows), "robust_variant"),
        "dropout_handle": (any(int(row.get("dropout_handle_substitutions", 0)) > 0 for row in rows), "dropout_handle_substitutions"),
        "friend_substitute": (any(int(row.get("friend_substitutions", 0)) > 0 for row in rows), "friend_substitutions"),
        "select_clients": (any(row.get("selected_by_action") for row in rows), "selected_by_action"),
        "spawn_concept_auto": (any(int(row.get("feddrift_num_models", 1)) > 1 for row in rows), "feddrift_num_models"),
    }
    observed, _ = checks[action]
    return observed, ACTION_MECHANISM_SIGNAL[action]


def result_row(job: dict, manifest: dict, cfg, engine: Engine, event_round: int,
               smoke: bool, code_sha: str, git_dirty: bool, status_sha: str) -> dict:
    set_seed(job["seed"])
    agent = FixedActionAgent(job["action"], event_round)
    started = time.perf_counter()
    hist = engine.run(agent, log=False)
    duration = time.perf_counter() - started

    post_acc = hist["acc"][event_round:]
    post_worst = hist.get("worst_acc", [])[event_round:]
    emitted = any(action == job["action"] for _, action in hist["actions"])
    mechanism_ok, signal = mechanism_result(job["action"], hist, event_round)
    tool = hist.get("tool", "none")
    activated = tool == job["action"] if job["action"] != "no_op" else tool == "none"
    row = {
        "run_id": job["run_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "code_sha": code_sha,
        "source_sha256": source_sha256(),
        "git_dirty": str(git_dirty).lower(),
        "git_status_sha256": status_sha,
        "dataset": cfg.dataset,
        "partition_mode": cfg.partition_mode,
        "smoke": str(smoke).lower(),
        "cause": job["cause"],
        "scenario": job["scenario"],
        "drift_type": job["drift_type"] or "n/a",
        "seed": job["seed"],
        "requested_action": job["action"],
        "effective_tool": tool,
        "action_emitted": str(emitted).lower(),
        "activation_observed": str(activated).lower(),
        "mechanism_observed": str(mechanism_ok).lower(),
        "mechanism_signal": signal,
        "event_round": event_round,
        "rounds": cfg.rounds,
        "post_event_mean_accuracy": round(statistics.fmean(post_acc), 8),
        "post_event_worst_class_accuracy": round(statistics.fmean(post_worst), 8),
        "final_accuracy": round(hist["acc"][-1], 8),
        "switch_round": hist["switch_round"] if hist["switch_round"] is not None else "n/a",
        "duration_seconds": round(duration, 3),
    }
    row.update(feddrift_structure_summary(hist.get("telemetry", ()), event_round))
    return row


def run_plan(plan: dict, manifest: dict, raw_path: Path, synthetic: bool, smoke: bool,
             out_dir: Path, resume: bool) -> None:
    existing: set[str] = set()
    code_sha, dirty, status_sha = git_metadata(Path(__file__).resolve().parent)
    current_source_sha = source_sha256()
    if raw_path.exists():
        if not resume:
            raise FileExistsError(f"{raw_path} exists; use --resume or choose another --out-dir")
        with raw_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(
            row.get("code_sha") != code_sha
            or row.get("source_sha256") != current_source_sha
            or row.get("git_dirty") != str(dirty).lower()
            or row.get("git_status_sha256") != status_sha
            for row in rows
        ):
            raise ValueError(
                f"resume identity mismatch in {raw_path}; archive it and restart this grid"
            )
        existing = {row["run_id"] for row in rows}

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if raw_path.exists() else "w"
    with raw_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if mode == "w":
            writer.writeheader()
        current_group = None
        cfg = engine = event_round = None
        for index, job in enumerate(plan["jobs"], start=1):
            if job["run_id"] in existing:
                continue
            group = (job["cause"], job["seed"])
            if group != current_group:
                cfg = make_cfg(
                    scenario=job["scenario"], seed=job["seed"], synthetic=synthetic,
                    smoke=smoke, formal_evidence=True, proxy_evidence=True,
                    drift_type=job["drift_type"], out_dir=str(out_dir), decide_every=1,
                    partition_mode=plan["partition_mode"],
                )
                if "hetero_beta" in plan.get("config_overrides", {}):
                    cfg.hetero_beta = float(plan["config_overrides"]["hetero_beta"])
                if "dropout_beta" in plan.get("config_overrides", {}):
                    cfg.dropout_beta = float(plan["config_overrides"]["dropout_beta"])
                    cfg.dirichlet_beta = cfg.dropout_beta
                if "dropout_cache_ttl" in plan.get("config_overrides", {}):
                    cfg.dropout_cache_ttl = int(
                        plan["config_overrides"]["dropout_cache_ttl"]
                    )
                event_round = 0 if job["scenario"] == "hetero" else cfg.drift_round
                set_seed(job["seed"])
                engine = Engine(cfg, build_data(cfg))
                current_group = group
            row = result_row(
                job, manifest, cfg, engine, event_round, smoke, code_sha, dirty, status_sha
            )
            writer.writerow(row)
            handle.flush()
            print(
                f"CALIBRATION_RUN {index}/{plan['job_count']}"
                f" cause={job['cause']} seed={job['seed']} action={job['action']}"
                f" utility={row['post_event_mean_accuracy']} mechanism={row['mechanism_observed']}"
            )


def aggregate_matrix(manifest: dict, raw_path: Path, matrix_path: Path,
                     partition_mode: str = "noniid") -> dict:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    calibration = manifest["calibration"]
    expected = build_plan(
        manifest, calibration["causes"], calibration["actions"], calibration["seeds"],
        dataset="cifar10", smoke=False, partition_mode=partition_mode,
    )
    expected_ids = {job["run_id"] for job in expected["jobs"]}
    actual_ids = [row["run_id"] for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("raw calibration contains duplicate run_id values")
    if set(actual_ids) != expected_ids:
        missing = len(expected_ids - set(actual_ids))
        extra = len(set(actual_ids) - expected_ids)
        raise ValueError(
            f"full CIFAR10 matrix requires {len(expected_ids)} exact jobs"
            f" (missing={missing}, extra={extra})"
        )
    if any(row["manifest_sha256"] != manifest_sha256(manifest) for row in rows):
        raise ValueError("raw calibration manifest SHA mismatch")
    if any(row.get("partition_mode") != partition_mode for row in rows):
        raise ValueError("raw calibration partition mode mismatch")

    cause_rows = aggregate_cause_rows(manifest, rows)
    epsilon = float(calibration["epsilon"])

    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    code_shas = sorted({row["code_sha"] for row in rows})
    source_shas = sorted({row["source_sha256"] for row in rows})
    matrix = {
        "schema_version": manifest["schema_version"],
        "status": "frozen",
        "manifest_sha256": manifest_sha256(manifest),
        "source_raw_file": raw_path.name,
        "source_raw_sha256": raw_sha,
        "code_shas": code_shas,
        "source_sha256": source_shas[0] if len(source_shas) == 1 else "mixed",
        "git_dirty_any": any(row["git_dirty"] == "true" for row in rows),
        "dataset": "cifar10",
        "partition_mode": partition_mode,
        "seeds": calibration["seeds"],
        "primary_utility": calibration["primary_utility"],
        "epsilon": epsilon,
        "run_count": len(rows),
        "causes": cause_rows,
    }
    digest = hashlib.sha256(json.dumps(
        matrix, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    matrix["matrix_sha256"] = digest
    plan_path = raw_path.with_name("atomic_action_calibration_plan.json")
    if not plan_path.is_file():
        raise ValueError(f"required calibration plan is missing: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    trusted_sha, dirty, _ = git_metadata(Path(__file__).resolve().parent)
    if dirty:
        raise ValueError("formal matrix aggregation requires a clean trusted checkout")
    audit_calibration(
        manifest, plan, raw_path, matrix, root=Path(__file__).resolve().parent,
        expected_code_sha=trusted_sha,
    )
    write_json(matrix_path, matrix)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run", action="store_true", help="execute the selected jobs")
    parser.add_argument("--aggregate", type=Path, help="aggregate a complete formal raw CSV")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--causes", nargs="+")
    parser.add_argument("--actions", nargs="+")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/b3_composite_v2/calibration"))
    parser.add_argument("--matrix-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition-mode", choices=PARTITION_MODES, default="noniid")
    parser.add_argument("--hetero-beta", type=float,
                        help="override beta for an isolated Non-IID hetero-only control grid")
    parser.add_argument("--dropout-beta", type=float,
                        help="override partition beta for an isolated Non-IID dropout-only grid")
    parser.add_argument("--dropout-cache-ttl", type=int,
                        help="override cache TTL for an isolated dropout-only control grid")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    args.out_dir = partition_artifact_path(
        args.out_dir, manifest, args.partition_mode
    )
    if args.matrix_out:
        args.matrix_out = partition_artifact_path(
            args.matrix_out.parent, manifest, args.partition_mode
        ) / args.matrix_out.name
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid B3 manifest: " + "; ".join(errors))
    calibration = manifest["calibration"]

    if any(value is not None for value in (
            args.hetero_beta, args.dropout_beta, args.dropout_cache_ttl)) and args.aggregate:
        raise ValueError("control overrides cannot be used while aggregating the canonical matrix")

    if args.aggregate:
        require_formal_manifest(manifest)
        matrix_path = args.matrix_out or args.aggregate.with_name(calibration["matrix_output"])
        require_versioned_output(matrix_path, manifest, args.partition_mode)
        matrix = aggregate_matrix(
            manifest, args.aggregate, matrix_path, args.partition_mode
        )
        print(f"B3_DAMAGE_MATRIX_OK path={matrix_path} sha256={matrix['matrix_sha256']}")
        return 0

    causes = selected(calibration["causes"], args.causes, "causes")
    actions = selected(calibration["actions"], args.actions, "actions")
    seeds = selected(calibration["seeds"], args.seeds, "seeds")
    if args.hetero_beta is not None:
        if args.hetero_beta <= 0:
            raise ValueError("--hetero-beta must be positive")
        if args.partition_mode != "noniid" or causes != ["partial_participation_hetero"]:
            raise ValueError(
                "--hetero-beta is only valid for a Non-IID partial_participation_hetero-only grid"
            )
    if args.dropout_beta is not None:
        if args.dropout_beta <= 0:
            raise ValueError("--dropout-beta must be positive")
        if args.partition_mode != "noniid" or causes != ["dropout"]:
            raise ValueError("--dropout-beta is only valid for a Non-IID dropout-only grid")
    if args.dropout_cache_ttl is not None:
        if args.dropout_cache_ttl <= 0:
            raise ValueError("--dropout-cache-ttl must be positive")
        if causes != ["dropout"]:
            raise ValueError("--dropout-cache-ttl is only valid for a dropout-only grid")
    dataset = "synthetic" if args.synthetic else "cifar10"
    if dataset == "cifar10" and not args.smoke:
        require_formal_manifest(manifest)
        require_versioned_output(args.out_dir, manifest, args.partition_mode)
    plan = build_plan(
        manifest, causes, actions, seeds, dataset, args.smoke, args.partition_mode,
        args.hetero_beta, args.dropout_beta, args.dropout_cache_ttl,
    )
    plan_path = args.out_dir / "atomic_action_calibration_plan.json"
    write_json(plan_path, plan)
    print(
        f"B3_CALIBRATION_PLAN_OK path={plan_path} jobs={plan['job_count']}"
        f" dataset={dataset} smoke={str(args.smoke).lower()}"
    )
    if not args.run:
        return 0

    raw_path = args.out_dir / calibration["raw_output"]
    run_plan(plan, manifest, raw_path, args.synthetic, args.smoke, args.out_dir, args.resume)
    if plan["job_count"] == calibration["grid_size"] and dataset == "cifar10" and not args.smoke:
        matrix_path = args.matrix_out or args.out_dir / calibration["matrix_output"]
        matrix = aggregate_matrix(
            manifest, raw_path, matrix_path, args.partition_mode
        )
        print(f"B3_DAMAGE_MATRIX_OK path={matrix_path} sha256={matrix['matrix_sha256']}")
    elif args.smoke:
        print(f"B3_CALIBRATION_SMOKE_OK raw={raw_path} runs={plan['job_count']}")
    else:
        print(f"B3_CALIBRATION_CONTROL_OK raw={raw_path} runs={plan['job_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
