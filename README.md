# DynFL-Bench

**Runtime Diagnosis and Intervention under Hidden Composite Causes in Federated Learning**

DynFL-Bench is an executable benchmark for evaluating runtime controllers in federated learning. During training, hidden causes such as concept drift, Byzantine updates, and changing client participation may overlap and evolve. A controller sees only server-visible observations and a compact control context, then emits a complete bundle of executable interventions.

The benchmark evaluates the decision itself—not final model accuracy alone. It asks whether a controller reaches an exact action state, reaches it in time, avoids irrelevant actions, and retains it as its own interventions change subsequent observations.

## What is evaluated

| Protocol | Coverage |
|---|---|
| Single cause | 7 cases spanning distribution shift, Byzantine updates, availability, participation, and staggered drift |
| Two causes | 9 compatible composite cases × 3 frozen test seeds = 27 episodes |
| Temporal | Sequential addition, three-cause addition/removal, recurrence, and repeated rotation |
| Cross-task | Federated keyword spotting on Speech Commands |

At round \(t\), the controller receives server-visible observations \(o_t=\phi(s_t)\) and control context \(m_t\), then returns an action bundle \(B_t=\pi(o_t,m_t)\). A bundle contains at most three compatible actions from distinct families, or `no_op` alone. Hidden event schedules, affected clients, injection mechanisms, and the frozen cause–action mapping are never exposed to the controller.

The observation interface contains:

- `G0`: round, horizon, decision cadence, and learning rate;
- `G1`: accuracy trends and client-loss statistics;
- `G2`: update norms, directional dispersion, group separation, and local-score changes;
- `G3`: input-moment changes and label-histogram distance;
- `G4`: participation, availability, online-set overlap, and selection utility;
- `G5`: active actions, duration, execution outcomes, and recent observation changes.

## Main result

DynFL-Bench separates three capabilities:

- **First hit:** Ever and Timely Exact Hit at 5/10 decisions (TEH@5/10).
- **Retention:** Ended-correct, Timely-and-End@10, post-hit retention (PHR), and first-regression delay (FRD).
- **Round-wise behavior:** Exact decision rate, over-intervention (OI), and stable-period false intervention (SFI).

On the 27 two-cause episodes, the calibrated Composite Rule reaches TEH@10 and ends correctly in every episode with 100% PHR. The three LLM agents reach an exact decision at least once in 55.6–66.7% of episodes in their principal low-effort settings, but end correctly in only 7.4–25.9%. Their round-wise exact rate remains 12.6–24.6%, while OI ranges from 53.7–77.5%.

Trajectory audits expose the recurring failure: once an intervention suppresses a symptom, the controller interprets the recovered observations as evidence that the underlying cause has disappeared and withdraws the action. DynFL-Bench therefore distinguishes transient diagnosis from sustained closed-loop control.

Full paper-aligned tables are available on the [project page](docs/index.html).

## Quick start

DynFL-Bench is tested with Python 3.12. For a CPU-only contract check:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tests/test_agent_decision_audit.py
python tests/test_b3_phase2.py
```

Run a synthetic smoke episode without downloading a dataset:

```bash
python src/run.py --smoke --synthetic --scenario drift --no_plot --out_dir results/smoke
```

Full CIFAR-10 and Speech Commands evaluations are intended for a CUDA-capable environment.

## Controller interface

```python
from agents import Agent
from action_contract import Action, ActionBundle

class MyController(Agent):
    name = "my_controller"

    def decide(self, observation):
        if (observation.get("participation_gap") or 0) > 0:
            return Action("dropout_handle")
        return ActionBundle(("no_op",))
```

The public observation contract is defined in `src/observation_contract.py`. The exact frozen LLM system prompt is `B3_SYSTEM_PROMPT` in `src/llm_backend.py`; its SHA-256 is recorded below. Action families, bundle compatibility, and lifecycle semantics are defined in `src/action_contract.py`.

## Official random baseline

`RandomBundleV2Agent` is the paper-facing random lower bound. At each decision it intervenes with probability `0.1`; conditional on intervention, it samples a bundle size uniformly from `{1, 2, 3}` and chooses compatible actions from distinct families.

```bash
python scripts/run_random_bundle_v2.py --suite two-cause --check
python scripts/run_random_bundle_v2.py --suite single-cause --check
```

Both suites use the same policy implementation. The older `RandomAgent` remains only for compatibility with early workflows and is not a formal paper baseline.

## Temporal protocols

```bash
python experiments/lifecycle/b3_lifecycle.py --protocol three_cause --check
python experiments/lifecycle/b3_lifecycle.py --protocol recurrence --check
python scripts/run_temporal_composition_ablation.py --case real_fault --order forward --seed 0 --agent rule --check
```

## Reproducibility

- CIFAR-10, a two-layer CNN, and 20 clients.
- Test seeds: `0`, `44`, and `56`; calibration seeds: `101`, `202`, and `303`.
- 80 rounds with one decision per round and a default event at round 40.
- Evaluated API agents: Qwen3.8-Flash (`none`/`low`/`xhigh`, temperature `0.6`), GLM-5.2 (`low`, temperature `0.0`), and DeepSeek-V4-Flash-0731 (`low`, temperature `0.6`).
- Paper-aligned result table: [`public_results/summary.csv`](public_results/summary.csv).
- Redacted episode, temporal, ablation, Speech Commands, provenance, and checksum artifacts: [`public_results/`](public_results/README.md).
- Frozen main manifest SHA-256: `af5afd16150f4378edeace0ad00151904c7fb8c6d15e8bcebb9ee9a8e3eb540d`.
- Frozen prompt SHA-256: `01779b937baf079ebc7b9c37a757e70bc0f40f83e784170b15850fa54f5ae291`.

API-backed controllers use OpenAI-compatible endpoints and read credentials only from environment variables. Copy `.env.example` for supported variable names; never commit a populated environment file.

Verify the frozen public evidence package without datasets, CUDA, or API credentials:

```bash
python scripts/build_public_results.py --verify
```

## Repository layout

```text
DynFL-Bench/
├── src/             benchmark runtime, agents, audit, scoring, and manifests
├── tests/           contract, resilience, engine, and audio frontend checks
├── scripts/         formal evaluation and result-audit entry points
├── experiments/     lifecycle protocols and batch jobs
├── docs/            static project page
└── public_results/  paper-aligned results, provenance, and checksums
```

The repository root intentionally contains only the project description, license, dependency list, environment template, and the directories above.

## License

DynFL-Bench is released under the [Apache License 2.0](LICENSE). Dataset licenses and terms remain with their respective providers.
