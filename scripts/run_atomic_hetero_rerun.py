"""Run the corrected atomic-heterogeneity table: 3 models x 3 seeds."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
RUNNER = ROOT / "scripts" / "run_atomic_llm.py"
from b3_manifest import DEFAULT_MANIFEST, load_manifest, manifest_sha256
from scripts.run_atomic_llm import FROZEN_MANIFEST_SHA
MODELS = {
    "qwen": ("qwen3.8-flash", 0.6),
    "glm": ("glm-5.2", 0.0),
    "deepseek": ("deepseek-v4-flash-0731", 0.6),
}
SEEDS = (0, 44, 56)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument(
        "--out-root", type=Path,
        default=ROOT / "results" / "atomic_hetero_rerun_v1",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    jobs = [(name, seed) for name in args.models for seed in args.seeds]
    if args.check:
        subprocess.run([sys.executable, str(RUNNER), "--model", "qwen3.8-flash",
                        "--temperature", "0.6", "--case", "hetero", "--seed", "0",
                        "--check"], check=True, cwd=ROOT)
        assert len(jobs) == len(args.models) * len(args.seeds)
        print(f"ATOMIC_HETERO_RERUN_CHECK_OK jobs={len(jobs)}")
        return

    actual_sha = manifest_sha256(load_manifest(DEFAULT_MANIFEST))
    if actual_sha != FROZEN_MANIFEST_SHA:
        raise RuntimeError(
            f"run from the frozen checkout: expected manifest {FROZEN_MANIFEST_SHA}, got {actual_sha}"
        )

    failures = []
    for index, (name, seed) in enumerate(jobs, 1):
        model, temperature = MODELS[name]
        command = [
            sys.executable, str(RUNNER), "--model", model,
            "--reasoning-effort", "low", "--temperature", str(temperature),
            "--case", "hetero", "--seed", str(seed),
            "--out-dir", str(args.out_root / name),
        ]
        print(f"ATOMIC_HETERO_RERUN_START {index}/{len(jobs)} model={name} seed={seed}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            failures.append((name, seed, result.returncode))
            print(f"ATOMIC_HETERO_RERUN_FAILED model={name} seed={seed} code={result.returncode}", flush=True)
        else:
            print(f"ATOMIC_HETERO_RERUN_DONE model={name} seed={seed}", flush=True)
    if failures:
        raise SystemExit(f"completed all jobs; failures={failures}")
    print(f"ATOMIC_HETERO_RERUN_OK jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
