"""Collect resumable, per-round public observations for detector calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from agents import NoOpAgent
from b3_calibrate import CAUSE_CONFIG
from b3_composite import prepare_episode
from b3_fingerprint import config_snapshot, git_identity, source_sha256
from b3_manifest import (
    DEFAULT_MANIFEST,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)
from engine import Engine
from multi_seed import make_cfg
from run import build_data, set_seed


TARGETED_COMPOSITE_CASES = ("real_dropout", "label_dropout")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return value


def make_config(cause: str, seed: int, out_dir: Path, partition_mode: str):
    scenario, drift_type = CAUSE_CONFIG[cause]
    return make_cfg(
        scenario=scenario,
        seed=seed,
        synthetic=False,
        smoke=False,
        formal_evidence=True,
        proxy_evidence=True,
        drift_type=drift_type,
        out_dir=str(out_dir),
        decide_every=1,
        partition_mode=partition_mode,
    )


def identity(manifest: dict, causes: list[str], seeds: list[int], partition_mode: str,
             composite_cases: list[str]) -> dict:
    configs = {
        cause: config_snapshot(make_config(cause, seeds[0], Path("<output>"), partition_mode))
        for cause in causes
    }
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": source_sha256(),
        "git": git_identity(),
        "dataset": "cifar10",
        "partition_mode": partition_mode,
        "agent": "noop",
        "api_calls": "disabled",
        "causes": causes,
        "composite_cases": composite_cases,
        "seeds": seeds,
        "job_count": (len(causes) + len(composite_cases)) * len(seeds),
        "stable_source": "rounds before each scenario event",
        "configs": configs,
    }


def resume_key(metadata: dict) -> dict:
    return {
        key: metadata[key]
        for key in (
            "manifest_sha256",
            "source_sha256",
            "dataset",
            "partition_mode",
            "agent",
            "api_calls",
            "causes",
            "composite_cases",
            "seeds",
            "job_count",
            "configs",
        )
    }


def append_run(path: Path, row: dict) -> None:
    fields = ["job", "cause", "seed", "status", "duration_seconds", "telemetry", "error"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def completed_jobs(telemetry_dir: Path, jobs: list[tuple[str, str, int]]) -> set[str]:
    return {
        f"{name}_s{seed}"
        for _, name, seed in jobs
        if (telemetry_dir / f"{name}_s{seed}.csv").is_file()
    }


def write_progress(path: Path, jobs: list[tuple[str, str, int]], done: set[str], durations: list[float],
                   failures: list[dict], status: str, current: str | None = None) -> None:
    mean = sum(durations) / len(durations) if durations else None
    remaining = len(jobs) - len(done)
    write_json(path, {
        "status": status,
        "updated_utc": utc_now(),
        "completed": len(done),
        "total": len(jobs),
        "remaining": remaining,
        "current": current,
        "mean_job_seconds": round(mean, 1) if mean is not None else None,
        "eta_seconds": round(mean * remaining) if mean is not None else None,
        "failures": failures,
    })


def write_telemetry(manifest: dict, history: dict, source: str, seed: int,
                    event_round: int, target: Path) -> None:
    observation_fields = list(manifest["public_observation_schema"])
    fields = ["source_cause", "target_cause", "seed", "event_round", *observation_fields]
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for telemetry in history["telemetry"]:
            round_id = int(telemetry["round"])
            row = {
                "source_cause": source,
                "target_cause": "stable" if round_id < event_round else source,
                "seed": seed,
                "event_round": event_round,
            }
            row.update({name: csv_value(telemetry[name]) for name in observation_fields})
            writer.writerow(row)
    tmp.replace(target)


def collect_one(manifest: dict, cause: str, seed: int, out_dir: Path,
                partition_mode: str, target: Path) -> float:
    cfg = make_config(cause, seed, out_dir, partition_mode)
    event_round = 0 if cfg.scenario == "hetero" else cfg.drift_round
    set_seed(seed)
    started = time.perf_counter()
    history = Engine(cfg, build_data(cfg)).run(NoOpAgent(), log=False)
    write_telemetry(manifest, history, cause, seed, event_round, target)
    return time.perf_counter() - started


def collect_composite_one(manifest: dict, case_id: str, seed: int, out_dir: Path,
                          partition_mode: str, target: Path, synthetic: bool = False,
                          smoke: bool = False) -> float:
    started = time.perf_counter()
    plan_path = out_dir / "private_plans" / f"{case_id}_s{seed}.json"
    plan_path.parent.mkdir(exist_ok=True)
    cfg, data, plan = prepare_episode(
        case_id, seed, synthetic=synthetic, smoke=smoke,
        plan_path=plan_path, resume=plan_path.exists(),
        partition_mode=partition_mode,
    )
    event_round = int(plan["hidden_event"]["event_schedule"][0]["round"])
    set_seed(seed)
    history = Engine(cfg, data).run(NoOpAgent(), log=False)
    write_telemetry(manifest, history, case_id, seed, event_round, target)
    return time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate and print the plan only")
    mode.add_argument("--run", action="store_true", help="execute the collection plan")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("results/detector_calibration_v1"))
    parser.add_argument("--partition-mode", choices=("iid", "noniid"), default="noniid")
    parser.add_argument("--causes", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int)
    composites = parser.add_mutually_exclusive_group()
    composites.add_argument("--include-targeted-composites", action="store_true",
                            help="also collect real_dropout and label_dropout for each seed")
    composites.add_argument("--composite-cases", nargs="+",
                            help="collect only the named composite cases in addition to --causes")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))

    available = list(manifest["calibration"]["causes"])
    causes = available if args.causes is None else args.causes
    unknown = set(causes) - set(available)
    if unknown:
        raise ValueError(f"unknown causes: {sorted(unknown)}")
    seeds = args.seeds or list(manifest["calibration"]["seeds"])
    composite_cases = list(
        args.composite_cases or
        (TARGETED_COMPOSITE_CASES if args.include_targeted_composites else ())
    )
    available_cases = {case["case_id"] for case in manifest["formal_cases"]}
    if unknown_cases := set(composite_cases) - available_cases:
        raise ValueError(f"unknown composite cases: {sorted(unknown_cases)}")
    jobs = ([('atomic', cause, seed) for cause in causes for seed in seeds] +
            [('composite', case_id, seed) for case_id in composite_cases for seed in seeds])
    metadata = identity(
        manifest, causes, seeds, args.partition_mode, composite_cases
    )

    print(
        f"PLAN jobs={len(jobs)} dataset=cifar10 partition={args.partition_mode} "
        f"seeds={','.join(map(str, seeds))} causes={','.join(causes)} "
        f"composites={','.join(composite_cases) or 'none'} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
        f"api_calls=disabled",
        flush=True,
    )
    if args.check:
        assert len(jobs) == (len(causes) + len(composite_cases)) * len(seeds)
        assert all(cause in CAUSE_CONFIG for cause in causes)
        assert all(case_id in available_cases for case_id in composite_cases)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = args.out_dir / "telemetry"
    telemetry_dir.mkdir(exist_ok=True)
    metadata_path = args.out_dir / "run_metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(f"{metadata_path} exists; pass --resume or choose another output")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if resume_key(previous) != resume_key(metadata):
            raise ValueError("resume identity mismatch; use the original code/settings or a new output directory")
    else:
        write_json(metadata_path, metadata)

    progress_path = args.out_dir / "progress.json"
    runs_path = args.out_dir / "runs.csv"
    done = completed_jobs(telemetry_dir, jobs)
    durations: list[float] = []
    failures: list[dict] = []
    write_progress(progress_path, jobs, done, durations, failures, "running")

    for index, (kind, name, seed) in enumerate(jobs, start=1):
        job = f"{name}_s{seed}"
        target = telemetry_dir / f"{job}.csv"
        if job in done:
            print(f"SKIP {index}/{len(jobs)} {job}", flush=True)
            continue
        write_progress(progress_path, jobs, done, durations, failures, "running", job)
        print(f"START {index}/{len(jobs)} {job}", flush=True)
        try:
            collector = collect_one if kind == "atomic" else collect_composite_one
            duration = collector(
                manifest, name, seed, args.out_dir, args.partition_mode, target
            )
            durations.append(duration)
            done.add(job)
            append_run(runs_path, {
                "job": job,
                "cause": name,
                "seed": seed,
                "status": "completed",
                "duration_seconds": round(duration, 3),
                "telemetry": str(target),
                "error": "",
            })
            print(f"DONE {index}/{len(jobs)} {job} seconds={duration:.1f}", flush=True)
        except Exception as exc:  # keep an overnight batch moving; --resume retries missing jobs
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"job": job, "error": error})
            append_run(runs_path, {
                "job": job,
                "cause": name,
                "seed": seed,
                "status": "failed",
                "duration_seconds": "",
                "telemetry": str(target),
                "error": error,
            })
            print(f"FAILED {index}/{len(jobs)} {job}: {error}", file=sys.stderr, flush=True)
            traceback.print_exc()
        write_progress(progress_path, jobs, done, durations, failures, "running")

    status = "completed" if len(done) == len(jobs) else "completed_with_failures"
    write_progress(progress_path, jobs, done, durations, failures, status)
    print(f"FINISH status={status} completed={len(done)}/{len(jobs)}", flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
