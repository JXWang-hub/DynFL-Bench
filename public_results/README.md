# Public result evidence

This directory is the compact, redacted evidence package for the paper. It is generated from the frozen private result archive; it does not contain datasets, checkpoints, raw API payloads, credentials, private hostnames, or absolute machine paths.

## Files

| File | Purpose |
|---|---|
| `summary.csv` | Paper table for the 7 single-cause and 9 two-cause protocols |
| `core_episode_results.csv` | Per-case, per-seed event outcomes and two-cause retention audits |
| `temporal_results.csv` | Sequential addition, removal, recurrence, and rotation metrics |
| `observation_ablation.csv` | G1--G4 TEH@10 ablations and G5 retention analysis |
| `speech_results.csv` | Speech Commands decision audit and intervention-value results |
| `mechanism_validation.csv` | Calibration-seed cause--action mechanism checks |
| `provenance.json` | Paper-to-artifact mapping, frozen identifiers, source hashes, and data classification |
| `run_integrity.json` | Export assertions and row-count checks |
| `SHA256SUMS.csv` | Size and SHA-256 for every public result artifact |

## Metric conventions

- `Ever`: at least one exact decision in an event interval.
- `TEH@5/10`: first exact decision within 5/10 decision opportunities.
- `Ended-correct`: the final decision in the interval is exact.
- `PHR`: macro-average exact-decision fraction from the first hit to the interval end, over hit intervals only.
- `FRD`: Kaplan--Meier median delay to the first post-hit regression; `NR` means the median was not reached.
- `Exact`: exact decisions divided by all planned event decision slots; slots lost to terminal failure count as incorrect.
- `OI`: decisions containing an irrelevant action divided by observed scorable event decisions.
- `SFI`: interventions during stable periods divided by observed stable-period decisions.

The observed-decision denominator for OI matters when an episode terminates early. For example, the two-cause Random Bundle v2 result is `69/960 = 7.2%`; dividing by the 1080 planned slots would use the wrong denominator.

## Build and verify

Maintainers can rebuild the package from a verified private archive:

```bash
python scripts/build_public_results.py --archive /path/to/finally_result
```

Anyone can verify the committed package without datasets, CUDA, or API credentials:

```bash
python scripts/build_public_results.py --verify
```

The public episode table intentionally contains derived audit outcomes rather than prompts, model responses, or full telemetry. Frozen API outputs are reused for the reported numbers; rerunning an API-backed experiment may vary with provider-side model changes.
