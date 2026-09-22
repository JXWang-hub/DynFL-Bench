"""Central configuration for DynFL-Bench (B1).

v1.1 taxonomy: a `scenario` switch demonstrates diagnosis-then-act.
  - scenario="drift": genuine concept drift. ALL clients relabel at drift_round;
    evaluation is on the NEW concept (does the model follow the drift?).
    Correct response = adapt (lr_reset). robust aggregation is the WRONG tool.
  - scenario="fault": faulty/poisoning clients (Phase A). A subset relabel;
    evaluation stays on the ORIGINAL task. Correct response = robust (reject).
  - scenario="dropout": clients drop out; eval on stable task. Correct = dropout_handle.
  - scenario="hetero": static extreme non-IID (low-beta Dirichlet, NO injection
    point; symptom = persistent high client divergence, NOT a sudden event).
    Eval on original task + worst-class accuracy. Correct response = Oort-supported
    select_clients under partial participation; FedProx remains a wrong-action ablation.
"""
from dataclasses import dataclass
from pathlib import Path

# Per-scenario fraction of affected clients — single source (run.py/multi_seed.py).
DRIFT_FRACTION = {"drift": 1.0, "fault": 0.2, "dropout": 0.0, "hetero": 0.0, "staggered": 0.0}
DATASETS = ("cifar10", "speechcommands35", "synthetic")
PARTITION_MODES = ("iid", "noniid")


def validate_partition_mode(value: str) -> str:
    if value not in PARTITION_MODES:
        raise ValueError(f"partition_mode must be one of {PARTITION_MODES}")
    return value


def partition_output_dir(path, partition_mode: str) -> Path:
    """Put each statistical partition under its own artifact root."""
    mode = validate_partition_mode(partition_mode)
    path = Path(path)
    return path if path.name == mode else path / mode


@dataclass
class Config:
    # --- scenario (v1.1) ---
    scenario: str = "drift"          # "drift" | "fault" | "dropout"

    # --- data ---
    dataset: str = "cifar10"         # see DATASETS
    data_root: str = "./data"
    partition_mode: str = "noniid"    # "iid" | "noniid"
    num_clients: int = 20
    dirichlet_beta: float = 0.3
    dropout_beta: float = 0.05       # sharper split so dropped clients ~exclusively
                                     # own a class range -> dropout actually hurts
    dropout_mode: str = "correlated" # "correlated" (a class-cluster goes absent,
                                     # AC1.3.2) | "random" (each round drop alpha
                                     # frac at random, AC1.3.1) | "fdms_clustered"
                                     # (5 clusters, per-cluster churn; FL-FDMS main)
    dropout_rate: float = 0.5        # alpha: per-round random dropout rate, sweep
                                     # {0.3, 0.5, 0.7} (only used in mode="random")
    fdms_clusters: int = 5           # FL-FDMS clustered CIFAR protocol: 20 clients
                                     # grouped into 5 clusters, each tied to 2 labels.
    fdms_warmup_rounds: int = 1      # source FL-FDMS/FL-Stale use round 0 full participation.
    fdms_error_samples_per_round: int = 2
    dropout_cache_ttl: int = 5       # rounds; only accepted real updates enter cache
    dropout_recover_round: int = 0   # >0: dropout is a TEMPORARY outage window
                                     # [drift_round, recover_round); clients return
                                     # after. 0 = no recovery (permanent absence /
                                     # per-round churn, the original behavior).
                                     # dropout_handle self-disables on recovery
                                     # (substitution is gated on absence), so no
                                     # explicit undo action is needed.
    samples_per_client: int = 800
    test_size: int = 2000

    # --- training ---
    rounds: int = 80
    local_epochs: int = 2
    batch_size: int = 32
    lr0: float = 0.05                # start LR (decays so lr_reset is meaningful)
    lr_decay: float = 0.97           # per-round exponential decay
    lr_floor: float = 0.005          # LR cannot decay below this
    participation: float = 1.0

    # --- drift / fault (AC1.2.1 sudden / AC1.2.2 gradual) ---
    drift_round: int = 40
    # drift_fraction is set per-scenario in run.py: drift=1.0 (all), fault=0.2
    drift_fraction: float = 1.0
    drift_mode: str = "sudden"       # "sudden" (all flip at drift_round)
                                     # | "gradual" (waves of p% every K rounds)
                                     # | "recurrent" (drift away then back, AC1.2.3)
    drift_wave_frac: float = 0.2     # p: fraction of clients per wave (PRD default 20%)
    recur_gap: int = 20              # recurrent: drift away @drift_round, back to the
                                     # original concept @drift_round+recur_gap (Flash)
    # --- drift SOURCE type (AC1.2.5, FedDAA): only for scenario="drift" ---
    drift_type: str = "real"         # real P(y|x) | virtual P(x) | label P(y)
    virtual_shift: float = 2.0       # virtual: inputs += this constant (P(x) shifts, y|x same)
    label_skew_frac: float = 0.5     # legacy label-drift knob; B1 uses long-tail below
    label_imbalance_factor: float = 20.0
    label_min_per_class: int = 2
    label_prior_strength: float = 0.10
    # staggered drift (AC1.2.4, FedDrift): clients split into n_concepts clusters,
    # each drifts to its OWN concept at a staggered time -> single model torn apart,
    # correct response = spawn_concept (per-concept models).
    n_concepts: int = 2
    stagger_gap: int = 20            # rounds between successive clusters drifting
    drift_wave_every: int = 5        # K: rounds between waves. PRD's K=100 targets
                                     # multi-thousand-round runs; scaled to 5 here
                                     # (80 rounds, drift@40 -> 5 waves: 40/45/50/55/60,
                                     # all flipped by r60, 20 rounds left to converge),
                                     # same 等比缩放 as drift_round 2000->40 (§11)
    feddrift_loss_delta: float = 0.05
    feddrift_warmup_rounds: int = 5
    feddrift_confirm_rounds: int = 2
    feddrift_cluster_threshold: float = 0.35
    feddrift_min_cluster_size: int = 2
    feddrift_max_concepts: int = 4
    feddrift_h_delta: float = 0.06
    feddrift_h_deltap: float = 0.06
    feddrift_mark_rounds: int = 2
    feddrift_history_decay: float = 0.9
    feddrift_history_cap: float = 5.0
    # ponytail: speed-mode FedDrift accuracy matrix. 128 is the time-saving
    # version; set to 0 for full source-style client dataloader evaluation.
    feddrift_eval_samples: int = 128
    feddaa_T: int = 6
    feddaa_tol_split: float = 0.35
    feddaa_tol_merge: float = 0.08
    feddaa_max_clusters: int = 6
    feddaa_min_cluster_size: int = 2
    feddaa_soft_temp: float = 1.0
    feddaa_label_weight: float = 1.0
    feddaa_history_momentum: float = 0.8
    fedlc_tau: float = 1.0              # PFLlib FedLC default: tau * class_count^(-1/4)

    # --- heterogeneity (hetero scenario, AC F1.5) ---
    hetero_beta: float = 0.01        # extreme low-beta Dirichlet -> persistent high
                                     # client divergence (FedProx's home turf)
    hetero_local_epochs: int = 15    # FedProx only helps when local drift is real:
                                     # more local steps + extreme non-IID = the client
                                     # drift FedAvg suffers from (FedProx paper setup).
                                     # local_epochs=2 is too few -> FedAvg stays stable.
    hetero_participation: float = 0.4
    oort_exploration_factor: float = 0.9
    oort_exploration_decay: float = 0.98
    oort_exploration_min: float = 0.1
    oort_round_threshold: float = 80.0
    oort_round_penalty: float = 2.0
    oort_pacer_step: int = 5
    oort_pacer_delta: float = 10.0
    oort_clip_bound: float = 0.95
    oort_sample_window: int = 5
    oort_blacklist_rounds: int = -1
    oort_blacklist_max_len: float = 0.5

    # --- aggregator ---
    robust_trim: float = 0.2         # trimmed-mean fraction dropped each end
    byzantine_frac: float = 0.2      # source-aligned compromised client fraction
    byzantine_scale: float = 1.5     # compromised clients upload -scale * local update
    robust_variant: str = "distance_trimmed_mean"  # yjlee22 trimmed_mean path
    fedprox_mu: float = 0.1          # FedProx proximal strength (set_aggregator(fedprox))

    # --- decision loop ---
    decide_every: int = 1
    formal_evidence: bool = False    # True: only paper-supported actions can be
                                     # counted as correct tools in scoring.
    proxy_evidence: bool = False     # True: B2 mode accepts supported + proxy actions
                                     # as stage-correct tools.

    # --- misc ---
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "./results"

    def smoke(self):
        self.num_clients = 6
        self.samples_per_client = 200
        self.test_size = 500
        self.rounds = 20
        self.drift_round = 10
        return self
