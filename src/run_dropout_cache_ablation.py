"""Run the six-job beta=0.05 dropout-cache TTL ablation."""

from __future__ import annotations

import argparse
import subprocess

from b3_calibrate import build_plan
from b3_manifest import DEFAULT_MANIFEST, load_manifest
from run_e01_calibration import ROOT, command, run_commands


TTLS = (5, 80)
ACTION = "dropout_handle"


def commands(manifest: dict) -> list[tuple[str, list[str]]]:
    root = ROOT.parent / manifest["result_root"]
    return [
        (
            f"TTL{ttl}",
            command(
                "noniid", root / f"dropout_beta_0p05_ttl{ttl}_ablation",
                "--causes", "dropout", "--actions", ACTION,
                "--dropout-beta", "0.05", "--dropout-cache-ttl", str(ttl),
            ),
        )
        for ttl in TTLS
    ]


def self_check(manifest: dict) -> None:
    calibration = manifest["calibration"]
    plans = [
        build_plan(
            manifest, ["dropout"], [ACTION], calibration["seeds"],
            "cifar10", False, "noniid", dropout_beta=0.05,
            dropout_cache_ttl=ttl,
        )
        for ttl in TTLS
    ]
    assert [plan["job_count"] for plan in plans] == [3, 3]
    assert [plan["config_overrides"] for plan in plans] == [
        {"dropout_beta": 0.05, "dropout_cache_ttl": 5},
        {"dropout_beta": 0.05, "dropout_cache_ttl": 80},
    ]
    assert not ({job["run_id"] for job in plans[0]["jobs"]}
                & {job["run_id"] for job in plans[1]["jobs"]})
    print("DROPOUT_CACHE_ABLATION_OK ttl5=3 ttl80=3 total=6")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate plans without training")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="concurrent TTL arms (default: 1; use 2 if memory permits)")
    args = parser.parse_args()
    manifest = load_manifest(DEFAULT_MANIFEST)
    self_check(manifest)
    if args.check:
        return 0

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout
    if status:
        raise RuntimeError("formal ablation requires a clean checkout; commit or stash changes first")

    run_commands(commands(manifest), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
