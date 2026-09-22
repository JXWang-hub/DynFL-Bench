"""Run the frozen two-model API queue after all non-API checkpoints exist."""

from run_b3_formal_eval import main


if __name__ == "__main__":
    raise SystemExit(main(fixed_stage="api"))
