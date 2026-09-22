"""Canonical content identities for B3 artifacts and resumable runs."""

from __future__ import annotations

import hashlib
import json
import dataclasses
import numbers
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def canonical_sha256(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_sha256(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_identity(root: Path = ROOT, require_clean: bool = False) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        sha = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("trusted Git checkout metadata is unavailable") from exc
    if require_clean and status:
        raise ValueError("formal run requires a clean trusted checkout")
    return {
        "git_sha": sha,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "bit_generator"):
        return _json_value(value.bit_generator.state)
    raise TypeError


def config_snapshot(cfg) -> dict:
    values = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        try:
            values[name] = _json_value(value)
        except TypeError:
            continue
    return values


def agent_snapshot(agent) -> dict:
    state = {}
    for name, value in vars(agent).items():
        try:
            state[name] = _json_value(value)
        except TypeError:
            continue
    query = getattr(agent, "query_fn", None)
    return {
        "class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "name": agent.name,
        "state": state,
        "llm_backend": (f"{getattr(query, '__module__', '')}."
                        f"{getattr(query, '__qualname__', '')}" if query else None),
        "llm_model": getattr(agent, "model_name", None),
        "llm_spec": _json_value(getattr(query, "backend_spec", {})) if query else None,
    }


def run_fingerprint(*, manifest_sha: str, plan: dict, cfg, data: dict, agent,
                    synthetic: bool, smoke: bool, prompt_sha: str) -> dict:
    if data.get("partition_mode") != cfg.partition_mode:
        raise ValueError("partition mode mismatch between Config and data")
    payload = {
        "benchmark_version": plan["benchmark_version"],
        "schema_version": plan["schema_version"],
        "manifest_sha256": manifest_sha,
        "plan_sha256": plan["plan_sha256"],
        "dataset": cfg.dataset,
        "partition_mode": cfg.partition_mode,
        "synthetic": synthetic,
        "smoke": smoke,
        "rounds": cfg.rounds,
        "config": config_snapshot(cfg),
        "partition_sha256": data["partition_sha256"],
        "source_sha256": source_sha256(),
        "agent": agent_snapshot(agent),
        "llm_prompt_sha256": prompt_sha,
    }
    return {"sha256": canonical_sha256(payload), **payload}
