"""One client-interval query for legacy scenarios and private B3 plans."""

from dataclasses import dataclass
import numpy as np


DATA_CAUSES = ("real_drift", "virtual_drift", "label_prior_drift")


@dataclass(frozen=True)
class EpisodeTimeline:
    causes: tuple[str, ...]
    primary_event_round: int
    intervals: dict[str, dict[int | None, tuple[tuple[int, int | None], ...]]]

    def has(self, cause: str) -> bool:
        return cause in self.causes

    def event_round(self) -> int:
        return self.primary_event_round

    def active(self, cause: str, round_idx: int, client_id: int | None = None) -> bool:
        by_client = self.intervals.get(cause, {})
        candidates = (interval for key, values in by_client.items()
                      if client_id is None or key is None or key == client_id
                      for interval in values)
        return any(start <= round_idx and (end is None or round_idx < end)
                   for start, end in candidates)

    def active_fraction(self, cause: str, round_idx: int, client_ids) -> float:
        ids = tuple(client_ids)
        return (sum(self.active(cause, round_idx, ci) for ci in ids) / len(ids)
                if ids else 0.0)


def _intervals_for_clients(clients, start, end=None):
    return {int(client): ((int(start), None if end is None else int(end)),)
            for client in clients}


def compile_fdms_dropout_intervals(num_clients, groups, dropout_rate, seed,
                                   start_round, total_rounds, warmup_rounds=0,
                                   recover_round=0):
    """Compile FDMS per-round churn once; callers only query the result."""
    normalized = [tuple(map(int, group)) for group in groups if group]
    flat = [client for group in normalized for client in group]
    if sorted(flat) != list(range(num_clients)):
        raise ValueError("fdms_groups must partition every client exactly once")
    if not 0.0 <= float(dropout_rate) <= 1.0:
        raise ValueError("dropout_rate must be in [0, 1]")
    stop = int(recover_round) if int(recover_round) > 0 else int(total_rounds)
    first = max(int(start_round), int(warmup_rounds))
    absent_rounds = {client: [] for client in range(num_clients)}
    target = max(1, int(round((1.0 - float(dropout_rate)) * num_clients)))
    for round_idx in range(first, stop):
        rng = np.random.default_rng(int(seed) + 2000 + round_idx)
        mins = [1 for _ in normalized]
        maxs = [len(group) for group in normalized]
        counts = None
        for _ in range(1000):
            draw = [int(rng.integers(lo, hi + 1)) for lo, hi in zip(mins, maxs)]
            if sum(draw) == target:
                counts = draw
                break
        if counts is None:
            counts = [min(hi, max(lo, int(round(target / len(normalized)))))
                      for lo, hi in zip(mins, maxs)]
        present = []
        for group, count in zip(normalized, counts):
            present.extend(rng.choice(group, count, replace=False).tolist())
        pool = [client for client in range(num_clients) if client not in present]
        while len(present) < target and pool:
            pick = int(rng.choice(pool))
            pool.remove(pick)
            present.append(pick)
        while len(present) > target:
            present.pop(int(rng.integers(0, len(present))))
        present = set(present)
        for client in range(num_clients):
            if client not in present:
                absent_rounds[client].append(round_idx)

    intervals = {}
    for client, rounds in absent_rounds.items():
        spans = []
        for round_idx in rounds:
            if spans and spans[-1][1] == round_idx:
                spans[-1] = (spans[-1][0], round_idx + 1)
            else:
                spans.append((round_idx, round_idx + 1))
        if spans:
            intervals[client] = tuple(spans)
    return intervals


def compile_episode_timeline(cfg, data: dict) -> EpisodeTimeline:
    private = data.get("b3_episode_plan")
    if private:
        hidden = private["hidden_event"]
        causes = tuple(hidden["active_causes"])
        primary = min(int(item["round"]) for item in hidden["event_schedule"])
        if "cause_intervals" in hidden:
            intervals = {
                cause: {
                    int(client): tuple((int(start), None if end is None else int(end))
                                       for start, end in spans)
                    for client, spans in by_client.items()
                }
                for cause, by_client in hidden["cause_intervals"].items()
            }
        else:
            intervals = {}
            for cause in causes:
                target = hidden["affected_clients"].get(cause, {})
                clients = target.get("clients")
                start = 0 if cause == "partial_participation_hetero" else primary
                end = (target.get("recover_round") or None) if cause == "dropout" else None
                intervals[cause] = (_intervals_for_clients(clients, start, end)
                                    if clients is not None else {None: ((start, end),)})
        return EpisodeTimeline(causes, primary, intervals)

    scenario = cfg.scenario
    causes = ({
        "drift": ({"real": "real_drift", "virtual": "virtual_drift",
                   "label": "label_prior_drift"}[cfg.drift_type],),
        "fault": ("fault",),
        "dropout": ("dropout",),
        "hetero": ("partial_participation_hetero",),
    }.get(scenario, ()))
    primary = 0 if scenario == "hetero" else int(cfg.drift_round)
    intervals = {}
    for cause in causes:
        end = None
        if cause == "dropout" and int(cfg.dropout_recover_round) > 0:
            end = int(cfg.dropout_recover_round)
        if cause in DATA_CAUSES and cfg.drift_mode == "recurrent":
            end = int(cfg.drift_round + cfg.recur_gap)
        if cause in DATA_CAUSES:
            starts = data.get("drift_schedule", {})
            intervals[cause] = {
                int(client): ((int(start), end),) for client, start in starts.items()
            }
        elif cause == "fault":
            intervals[cause] = _intervals_for_clients(
                data.get("fault_clients", ()), primary
            )
        elif cause == "dropout" and cfg.dropout_mode == "random":
            spans = {client: [] for client in range(len(data["client_X"]))}
            stop = int(cfg.dropout_recover_round) if int(cfg.dropout_recover_round) > 0 else int(cfg.rounds)
            for round_idx in range(primary, stop):
                count = int(cfg.dropout_rate * len(spans))
                rng = np.random.default_rng(cfg.seed + 1000 + round_idx)
                for client in rng.choice(len(spans), count, replace=False).tolist() if count else ():
                    spans[client].append((round_idx, round_idx + 1))
            intervals[cause] = {client: tuple(values) for client, values in spans.items() if values}
        elif cause == "dropout" and cfg.dropout_mode == "fdms_clustered":
            intervals[cause] = compile_fdms_dropout_intervals(
                len(data["client_X"]), data.get("fdms_groups", ()),
                cfg.dropout_rate, cfg.seed, primary, cfg.rounds,
                getattr(cfg, "fdms_warmup_rounds", 0), cfg.dropout_recover_round,
            )
        elif cause == "dropout":
            intervals[cause] = _intervals_for_clients(
                data.get("dropped_clients", ()), primary, end
            )
        else:
            intervals[cause] = {None: ((0, None),)}
    return EpisodeTimeline(causes, primary, intervals)
