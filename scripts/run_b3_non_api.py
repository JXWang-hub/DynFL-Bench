"""Run one resumable shard of the frozen non-API evaluation queue."""

from run_b3_formal_eval import main


if __name__ == "__main__":
    raise SystemExit(main(fixed_stage="deterministic"))
