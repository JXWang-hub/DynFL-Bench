"""Single source for scenario -> trigger -> action -> paper evidence."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    venue: str
    url: str
    claim: str
    repo_url: str = ""
    repo_note: str = ""


@dataclass(frozen=True)
class ActionEvidence:
    scenario: str
    subtype: str | None
    trigger: str
    action: str
    status: str          # algorithm_supported | source_aligned | proxy | baseline_only | unsupported
    papers: tuple[str, ...]
    current_impl: str
    deviations: tuple[str, ...]
    required_fix: str
    allowed_as_correct_tool: bool


PAPERS = {
    "feddrift": Paper(
        "feddrift",
        "Federated Learning under Distributed Concept Drift",
        "AISTATS 2023",
        "https://arxiv.org/abs/2206.00799",
        "Distributed drifts can be staggered in time and space; a single model is ill-suited.",
        "https://github.com/microsoft/FedDrift",
        "Official Microsoft source repository for the AISTATS 2023 FedDrift paper.",
    ),
    "feddaa": Paper(
        "feddaa",
        "FedDAA: Dynamic Client Clustering for Concept Drift Adaptation in Federated Learning",
        "arXiv 2025",
        "https://arxiv.org/abs/2506.21054",
        "Real, virtual, and label drift require different adaptation choices.",
        "https://github.com/LeegerPENG/FedDAA-Dynamic-Client-Clustering-for-Concept-Drift-Adaptation-in-Federated-Learning",
        "Author repository with CIFAR-10/CIFAR-100/Fashion-MNIST FedDAA entry scripts.",
    ),
    "fedlc": Paper(
        "fedlc",
        "Federated Learning with Label Distribution Skew via Logits Calibration",
        "ICML 2022",
        "https://proceedings.mlr.press/v162/zhang22p.html",
        "Logit calibration targets label-distribution skew in federated learning.",
        "https://github.com/TsingZ0/PFLlib",
        "PFLlib implementation: system/flcore/clients/clientlc.py and servers/serverlc.py.",
    ),
    "bbse": Paper(
        "bbse",
        "Detecting and Correcting for Label Shift with Black Box Predictors",
        "ICML 2018",
        "https://arxiv.org/abs/1802.03916",
        "Black-box predictors can estimate and correct target label priors under label shift.",
        "https://github.com/kundajelab/abstention",
        "Implements black-box label-shift correction utilities including BBSE-style estimators.",
    ),
    "mlls": Paper(
        "mlls",
        "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation",
        "arXiv 2019",
        "https://arxiv.org/abs/1901.06852",
        "Label-shift adaptation can correct predictions by estimating target priors without retraining the model.",
        "https://github.com/kundajelab/labelshiftexperiments",
        "Experiment repository; core correction utilities are in kundajelab/abstention.",
    ),
    "logit_adjust": Paper(
        "logit_adjust",
        "Long-tail learning via logit adjustment",
        "ICLR 2021",
        "https://arxiv.org/abs/2007.07314",
        "Adding the class-prior logit adjustment to cross-entropy addresses long-tailed label priors.",
        "",
        "No official minimal source repository found; implementation target is the paper loss.",
    ),
    "balms": Paper(
        "balms",
        "Balanced Meta-Softmax for Long-Tailed Visual Recognition",
        "NeurIPS 2020",
        "https://arxiv.org/abs/2007.10740",
        "Balanced Softmax adds the label-frequency prior inside the softmax loss for long-tailed data.",
        "https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification",
        "Official implementation for Balanced Softmax / Balanced Meta-Softmax.",
    ),
    "deep_coral": Paper(
        "deep_coral",
        "Deep CORAL: Correlation Alignment for Deep Domain Adaptation",
        "ECCV Workshops 2016",
        "https://arxiv.org/abs/1607.01719",
        "Moment alignment reduces domain/covariate shift by matching source and target feature statistics.",
        "https://github.com/VisionLearningGroup/CORAL",
        "Reference implementation for CORAL/Deep CORAL domain adaptation.",
    ),
    "fl_fdms": Paper(
        "fl_fdms",
        "Combating Client Dropout in Federated Learning via Friend Model Substitution",
        "arXiv 2022",
        "https://arxiv.org/abs/2205.13222",
        "Absent clients are substituted by updates from similar online friends.",
        "https://github.com/ystex/Federated-Learning-with-Friend-Discovery-and-Model-Substitution-",
        "Author repository for Friend Discovery and Model Substitution.",
    ),
    "mifa": Paper(
        "mifa",
        "Fast Federated Learning in the Presence of Arbitrary Device Unavailability",
        "NeurIPS 2021",
        "https://arxiv.org/abs/2106.04159",
        "MIFA corrects missing-client bias using memorized latest updates from unavailable devices.",
        "",
        "No public source repository found; implementation target is the paper algorithm.",
    ),
    "fedvarp": Paper(
        "fedvarp",
        "FedVARP: Tackling the Variance Due to Partial Client Participation in Federated Learning",
        "arXiv 2022",
        "https://arxiv.org/abs/2207.14130",
        "The server maintains each client's most recent update and uses it as a surrogate for non-participating clients.",
        "",
        "No public source repository found; implementation target is the paper algorithm.",
    ),
    "fedprox": Paper(
        "fedprox",
        "Federated Optimization in Heterogeneous Networks",
        "MLSys 2020",
        "https://arxiv.org/abs/1812.06127",
        "A proximal term improves robustness under systems and statistical heterogeneity.",
        "https://github.com/litian96/FedProx",
        "Official FedProx source repository.",
    ),
    "byz_trimmed_mean": Paper(
        "byz_trimmed_mean",
        "Security-Preserving Federated Learning via Byzantine-Sensitive Triplet Distance",
        "ISBI 2024",
        "https://arxiv.org/abs/2210.16519",
        "The source implements FedAvg, Krum, Trimmed-mean, Fang, and DCA Byzantine-robust baselines.",
        "https://github.com/yjlee22/byzantineFL",
        "Official repository for the ISBI 2024 implementation; its trimmed_mean is a distance-filtered FedAvg baseline, not ICML 2018 coordinate-wise trimmed mean.",
    ),
    "feddrl": Paper(
        "feddrl",
        "Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning",
        "ICPP 2022",
        "https://arxiv.org/abs/2208.02442",
        "Client impact factors can be adapted to non-IID data.",
        "",
        "No public source repository found; do not use as a strict implementation target.",
    ),
    "fedlmd": Paper(
        "fedlmd",
        "Federated Learning with Label-Masking Distillation",
        "ACM MM 2023",
        "https://doi.org/10.1145/3581783.3611984",
        "Label-masking distillation targets label-skewed federated learning.",
        "https://github.com/wnma3mz/FedLMD",
        "Official implementation; recommended open-source replacement target for the label-skew action.",
    ),
    "fedmgd": Paper(
        "fedmgd",
        "Modeling Global Distribution for Federated Learning with Label Distribution Skew",
        "Pattern Recognition 2023",
        "https://arxiv.org/abs/2212.08883",
        "A global generator models the global label-skewed distribution for FL.",
        "https://github.com/Sheng-T/FedMGD",
        "Open-source alternative, but heavier than FedLMD because it adds GAN training.",
    ),
    "oort": Paper(
        "oort",
        "Oort: Efficient Federated Learning via Guided Participant Selection",
        "OSDI 2021",
        "https://www.usenix.org/conference/osdi21/presentation/lai",
        "Guided participant selection improves FL training efficiency using statistical utility and system efficiency.",
        "https://github.com/SymbioticLab/Oort",
        "OSDI artifact repository; README notes Oort was later merged into FedScale.",
    ),
}


ACTION_EVIDENCE = (
    ActionEvidence(
        "stable", None,
        "No disturbance is detected.",
        "no_op", "baseline_only", (),
        "Do nothing.",
        (),
        "Keep as control baseline only.",
        False,
    ),
    ActionEvidence(
        "drift", "real",
        "Accuracy drops while most clients move together.",
        "lr_reset", "unsupported", (),
        "Reset current_lr to cfg.lr0 and keep one global model.",
        ("No verified FL concept-drift paper is implemented by this action.",),
        "Keep as heuristic baseline; replace formal correct action with FedDrift/FedDAA-style adaptation.",
        False,
    ),
    ActionEvidence(
        "drift", "real",
        "Real concept drift: P(y|x) changes.",
        "drift_adapt", "source_aligned", ("feddaa",),
        "FedDAA source-aligned dynamic client clustering using client R-by-R reports, prototype/silhouette splitting, per-learner weights, and loss-based routing.",
        ("Does not vendor the author runner/logger/data scripts.", "Uses a small NumPy k-means/silhouette implementation instead of the author script's sklearn runner path."),
        "Keep formal claims scoped to FedDAA core adaptation, not bitwise reproduction of the author scripts.",
        True,
    ),
    ActionEvidence(
        "drift", "virtual",
        "Input shift is high while label semantics remain stable.",
        "retain_history_adapt", "unsupported", ("feddaa",),
        "FedDAA-backed history-retaining cluster adaptation; clean clients can merge old/current data while shifted clients train on current data.",
        ("Poor fit for the current all-client same-direction virtual shift workpoint.", "Kept only for backward compatibility with older runs."),
        "Use moment_align_adapt for the current virtual drift scenario.",
        False,
    ),
    ActionEvidence(
        "drift", "virtual",
        "Input shift is high while label semantics remain stable.",
        "moment_align_adapt", "source_aligned", ("deep_coral",),
        "Align shifted inputs back to pre-drift global mean/std before local training and evaluation.",
        ("Uses scalar input mean/std rather than full CORAL covariance alignment.", "Estimates target moments from current shifted client tensors inside DynFL-Bench."),
        "Use full covariance or feature-layer CORAL only if scalar moments stop separating virtual drift.",
        True,
    ),
    ActionEvidence(
        "drift", "label",
        "Global label-prior shift is high.",
        "label_prior_adapt", "source_aligned", ("logit_adjust", "balms"),
        "Balanced-Softmax-style local loss: train with logits + log(pi_target) on label-drift rounds.",
        ("Uses a simulated secure-aggregation boundary over roster-matched baseline/current label reports; no cryptographic protocol is claimed.",),
        "Keep the claim scoped to report-derived prior adaptation, not target-label oracle access.",
        True,
    ),
    ActionEvidence(
        "drift", "label",
        "Client-level label skew is high.",
        "reweight", "source_aligned", ("fedlc",),
        "FedLC-style logit calibration: subtract tau * global_class_count^(-1/4) before cross-entropy on label-drift rounds.",
        ("Uses DynFL-Bench tensors instead of PFLlib's client/server runner.", "Kept as an A/B baseline for client label-skew rather than current same-direction prior drift."),
        "Use label_prior_adapt as the formal B1 answer for global same-direction label-prior drift.",
        False,
    ),
    ActionEvidence(
        "drift", "recurrent",
        "A concept drifts away and later returns.",
        "reuse_concept_model", "unsupported", ("feddrift", "feddaa"),
        "Not implemented.",
        ("Current recurrent response still relies on lr_reset-style adaptation.",),
        "Persist old concept models and reuse them on drift-back.",
        False,
    ),
    ActionEvidence(
        "fault", None,
        "A few clients show anomalous updates or poisoning.",
        "robust", "source_aligned", ("byz_trimmed_mean",),
        "Source-aligned yjlee22/byzantineFL trimmed_mean: distance-filter clients, then FedAvg.",
        ("Krum/Fang/DCA are not implemented.", "coord_trimmed_mean remains an ablation only.",
         "Full CIFAR long-run table is pending."),
        "Keep formal claims scoped to the yjlee22 trimmed_mean baseline.",
        True,
    ),
    ActionEvidence(
        "dropout", None,
        "Participation rate is below 1.0.",
        "dropout_handle", "algorithm_supported", ("mifa", "fedvarp"),
        "Substitute absent clients with their own last-seen update, matching MIFA/FedVARP stale-update memory.",
        ("Uses model-update/state substitution inside DynFL-Bench rather than reproducing the authors' optimizer loops.",),
        "Keep formal claims scoped to stale-update compensation for partial participation.",
        True,
    ),
    ActionEvidence(
        "dropout", None,
        "Participation rate is below 1.0.",
        "friend_substitute", "source_aligned", ("fl_fdms",),
        "Use the current update of the most update-similar online friend from the historical client-state matrix.",
        ("Author-runner reproduction is not vendored; full multi-seed CIFAR performance gate is pending.",),
        "Run the IID/Non-IID multi-seed gate before making runner-level or final performance claims.",
        True,
    ),
    ActionEvidence(
        "hetero", None,
        "Persistent high client divergence under non-IID data.",
        "fedprox", "source_aligned", ("fedprox",),
        "Local loss adds (mu/2)||w - w_global||^2.",
        ("Rolling trigger and harder hetero workpoint are implemented, but full multi-seed/CIFAR separation is pending.",),
        "Keep FedProx as a supported wrong-action ablation for the current partial-participation B1 workpoint.",
        False,
    ),
    ActionEvidence(
        "hetero", None,
        "Partial participation needs guided client sampling.",
        "select_clients", "source_aligned", ("oort",),
        "Oort-backed training selector with statistical utility, exploration/exploitation, deterministic system duration penalty, pacer, and blacklist.",
        ("Uses deterministic synthetic duration profiles instead of real device wall-clock traces.",
         "Does not implement Oort's testing selector or FedScale/Oort runner."),
        "Keep formal use scoped to hetero partial participation; use scenario-specific papers for dropout/drift/staggered.",
        True,
    ),
    ActionEvidence(
        "staggered", None,
        "Clients drift to different concepts at different times.",
        "spawn_concept", "proxy", ("feddrift",),
        "Clone K models and assign clients using ground-truth concept_of.",
        ("Uses ground-truth clusters.", "No automatic drift clustering.", "K is fixed."),
        "Implement spawn_concept_auto; keep truth clustering as oracle.",
        False,
    ),
    ActionEvidence(
        "staggered", None,
        "Clients drift to different concepts at different times.",
        "spawn_concept_auto", "source_aligned", ("feddrift",),
        "FedDrift source-aligned accuracy-matrix drift detection, post-activation assignment confirmation, stable-client handling, hierarchical merging, and capped dynamic multi-model routing.",
        ("Does not migrate the FedML/MPI/W&B runner.", "Uses DynFL-Bench's staggered benchmark protocol and round budget.",
         "Evaluates a deterministic client-data slice for speed instead of the full FedML dataloader matrix."),
        "Validate oracle gap on CIFAR/more seeds before claiming runner-level reproduction.",
        True,
    ),
    ActionEvidence(
        "any", None,
        "Lower-bound randomized baseline.",
        "random", "baseline_only", (),
        "RandomAgent samples implemented actions with p_act.",
        (),
        "Report over multiple seeds; never treat as a repair action.",
        False,
    ),
)


def _matches(e, scenario, action=None, subtype=None):
    scenario_ok = e.scenario in (scenario, "any")
    action_ok = action is None or e.action == action
    subtype_ok = subtype is None or e.subtype in (None, subtype)
    return scenario_ok and action_ok and subtype_ok


def evidence_for(scenario, action, subtype=None):
    return tuple(e for e in ACTION_EVIDENCE if _matches(e, scenario, action, subtype))


def correct_actions(scenario, subtype=None, formal=False, proxy=False):
    out = []
    for e in ACTION_EVIDENCE:
        if not _matches(e, scenario, subtype=subtype):
            continue
        if proxy:
            if e.status in ("algorithm_supported", "source_aligned", "proxy"):
                out.append(e.action)
        elif formal:
            if e.status in ("algorithm_supported", "source_aligned") and e.allowed_as_correct_tool:
                out.append(e.action)
        elif e.status in ("algorithm_supported", "source_aligned", "proxy"):
            out.append(e.action)
    return tuple(dict.fromkeys(out))


def all_actions():
    return tuple(sorted({e.action for e in ACTION_EVIDENCE}))


EVIDENCE_STATUSES = frozenset({
    "algorithm_supported", "source_aligned", "proxy", "baseline_only", "unsupported",
})

# Runtime telemetry field that proves each B3 action actually took effect.
ACTION_MECHANISM_SIGNAL = {
    "no_op": "no_active_tool",
    "drift_adapt": "using_drift_adapt",
    "moment_align_adapt": "moment_align_enabled",
    "label_prior_adapt": "label_prior_adapt_enabled",
    "robust": "robust_filtered_clients",
    "dropout_handle": "dropout_handle_substitutions",
    "friend_substitute": "friend_substitutions",
    "select_clients": "selected_by_action",
    "spawn_concept_auto": "feddrift_num_models",
}


def action_evidence_status(action):
    statuses = {e.status for e in ACTION_EVIDENCE if e.action == action}
    if len(statuses) != 1:
        raise ValueError(f"{action}: expected one evidence status, got {sorted(statuses)}")
    return statuses.pop()


def prompt_action_cards(formal=False):
    rows = []
    for e in ACTION_EVIDENCE:
        if e.status == "baseline_only":
            continue
        if formal and not e.allowed_as_correct_tool:
            continue
        paper = ", ".join(PAPERS[p].title for p in e.papers) or "no paper"
        rows.append(f"- {e.action}: {e.scenario}/{e.subtype or '*'}; {e.status}; {paper}. {e.required_fix}")
    return "\n".join(rows)


def validate_registry(actions=None, scenarios=None):
    actions = set(actions or all_actions())
    scenarios = set(scenarios or ("drift", "fault", "dropout", "hetero", "staggered", "stable"))
    missing_actions = sorted(a for a in actions if not any(e.action == a for e in ACTION_EVIDENCE))
    missing_scenarios = sorted(s for s in scenarios if not any(e.scenario == s for e in ACTION_EVIDENCE))
    bad_papers = sorted({p for e in ACTION_EVIDENCE for p in e.papers if p not in PAPERS})
    bad_statuses = sorted({e.status for e in ACTION_EVIDENCE if e.status not in EVIDENCE_STATUSES})
    if missing_actions or missing_scenarios or bad_papers or bad_statuses:
        raise AssertionError(
            f"missing_actions={missing_actions}, missing_scenarios={missing_scenarios}, "
            f"bad_papers={bad_papers}, bad_statuses={bad_statuses}"
        )


def write_markdown(path="results/evidence_report.md"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# DynFL-Bench Evidence Report\n\n")
        f.write("## Paper Registry\n\n")
        f.write("| key | paper | source_repo | note |\n")
        f.write("|---|---|---|---|\n")
        for p in PAPERS.values():
            repo = f"[repo]({p.repo_url})" if p.repo_url else "no public source found"
            note = p.repo_note or ""
            f.write(f"| `{p.key}` | [{p.title}]({p.url}) | {repo} | {note} |\n")
        f.write("\n")
        for e in ACTION_EVIDENCE:
            papers = ", ".join(f"[{PAPERS[p].title}]({PAPERS[p].url})" for p in e.papers) or "None"
            repos = ", ".join(
                f"[{PAPERS[p].repo_url}]({PAPERS[p].repo_url})" if PAPERS[p].repo_url else f"{PAPERS[p].title}: no public source found"
                for p in e.papers
            ) or "None"
            repo_notes = "; ".join(PAPERS[p].repo_note for p in e.papers if PAPERS[p].repo_note) or "None"
            deviations = "; ".join(e.deviations) or "None"
            f.write(f"## {e.scenario} / {e.subtype or '*'} / {e.action}\n\n")
            f.write(f"- status: `{e.status}`\n")
            f.write(f"- trigger: {e.trigger}\n")
            f.write(f"- papers: {papers}\n")
            f.write(f"- source_repos: {repos}\n")
            f.write(f"- repo_notes: {repo_notes}\n")
            f.write(f"- current_impl: {e.current_impl}\n")
            f.write(f"- deviations: {deviations}\n")
            f.write(f"- required_fix: {e.required_fix}\n")
            f.write(f"- allowed_as_correct_tool: {e.allowed_as_correct_tool}\n\n")
    return path


if __name__ == "__main__":
    validate_registry(
        actions=("no_op", "lr_reset", "drift_adapt", "retain_history_adapt", "moment_align_adapt", "robust",
                 "dropout_handle", "friend_substitute", "fedprox", "spawn_concept",
                 "spawn_concept_auto", "label_prior_adapt", "reweight", "select_clients"),
        scenarios=("drift", "fault", "dropout", "hetero", "staggered", "stable"),
    )
    path = write_markdown()
    print(f"OK: {len(ACTION_EVIDENCE)} evidence entries, report={path}")
