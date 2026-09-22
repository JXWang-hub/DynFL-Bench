"""Run the paired E01 calibration grids and the beta=0.3 hetero control."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from b3_calibrate import build_plan
from b3_manifest import DEFAULT_MANIFEST, load_manifest


ROOT = Path(__file__).resolve().parent
HETERO_CAUSE = "partial_participation_hetero"


def command(partition: str, out_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable, "-u", str(ROOT / "b3_calibrate.py"), "--run", "--resume",
        "--partition-mode", partition, "--out-dir", str(out_dir), *extra,
    ]


def commands(manifest: dict) -> list[tuple[str, list[str]]]:
    root = ROOT.parent / manifest["result_root"]
    return [
        ("IID", command("iid", root / "calibration")),
        ("NONIID", command("noniid", root / "calibration")),
        ("HETERO03", command(
            "noniid", root / "calibration_hetero_beta_0p3",
            "--causes", HETERO_CAUSE, "--hetero-beta", "0.3",
        )),
    ]


def self_check(manifest: dict) -> None:
    calibration = manifest["calibration"]
    canonical = build_plan(
        manifest, calibration["causes"], calibration["actions"], calibration["seeds"],
        "cifar10", False, "noniid",
    )
    control = build_plan(
        manifest, [HETERO_CAUSE], calibration["actions"], calibration["seeds"],
        "cifar10", False, "noniid", hetero_beta=0.3,
    )
    dropout_control = build_plan(
        manifest, ["dropout"], calibration["actions"], calibration["seeds"],
        "cifar10", False, "noniid", dropout_beta=0.05,
    )
    assert canonical["job_count"] == 189
    assert control["job_count"] == 27
    assert control["config_overrides"] == {"hetero_beta": 0.3}
    assert dropout_control["job_count"] == 27
    assert dropout_control["config_overrides"] == {"dropout_beta": 0.05}
    assert not ({job["run_id"] for job in canonical["jobs"]}
                & {job["run_id"] for job in control["jobs"]})
    print("E01_CALIBRATION_RUNNER_OK iid=189 noniid=189 hetero_beta_0p3=27")


def run_commands(items: list[tuple[str, list[str]]], workers: int) -> None:
    active = []
    active_lock = threading.Lock()
    output_lock = threading.Lock()

    def run_one(label: str, argv: list[str]) -> None:
        with output_lock:
            print(f"\n=== {label} ===", flush=True)
        process = subprocess.Popen(
            argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with active_lock:
            active.append(process)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                with output_lock:
                    print(f"[{label}] {line}", end="", flush=True)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, argv)
        finally:
            with active_lock:
                active.remove(process)

    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(run_one, *item) for item in items]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        with active_lock:
            for process in active:
                if process.poll() is None:
                    process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate plans without training")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2,
                        help="concurrent isolated calibration processes (default: 2)")
    args = parser.parse_args()
    manifest = load_manifest(DEFAULT_MANIFEST)
    self_check(manifest)
    if args.check:
        probes = [
            (f"CHECK{i}", [sys.executable, "-u", "-c", f"print('worker={i}')"])
            for i in range(1, 4)
        ]
        run_commands(probes, 2)
        print("E01_DUAL_PROCESS_OK workers=2 tasks=3")
        return 0

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout
    if status:
        raise RuntimeError("formal E01 requires a clean checkout; commit or stash changes first")

    run_commands(commands(manifest), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
