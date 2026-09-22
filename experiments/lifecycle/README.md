# Lifecycle experiments

These experiments extend the frozen benchmark with cause addition, removal, recurrence, and clearing. Shared observation, action, execution, and audit contracts live in `src/`.

Run fast protocol checks from the repository root:

```bash
python experiments/lifecycle/b3_lifecycle.py --protocol three_cause --check
python experiments/lifecycle/b3_lifecycle.py --protocol recurrence --check
```

The scripts in `jobs/` are optional server launchers for the frozen three-seed runs.
