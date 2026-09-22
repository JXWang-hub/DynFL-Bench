"""Minimal client/update runtime contract; stdlib-only."""

from dataclasses import dataclass, replace
import math
from typing import Any


@dataclass(frozen=True)
class UpdateEnvelope:
    client_id: int
    source_round: int
    source_kind: str
    origin_client: int
    trust_state: str
    state_dict: Any
    sample_count: int

    def trusted(self):
        return replace(self, trust_state="trusted")


@dataclass(frozen=True)
class FedDriftReport:
    client_id: int
    source_round: int
    sample_count: int
    model_scores: tuple[tuple[int, int, float], ...]

    def __post_init__(self):
        model_ids = [model_id for model_id, _, _ in self.model_scores]
        if (self.client_id < 0 or self.source_round < 0 or self.sample_count <= 0 or
                not self.model_scores or len(model_ids) != len(set(model_ids))):
            raise ValueError("invalid FedDrift report identity")
        if any(model_id < 0 or version < 0 or not math.isfinite(float(score))
               for model_id, version, score in self.model_scores):
            raise ValueError("invalid FedDrift model score")

    def scores(self):
        return {model_id: float(score) for model_id, _, score in self.model_scores}


@dataclass(frozen=True)
class LabelHistogramReport:
    client_id: int
    source_round: int
    class_counts: tuple[int, ...]

    def __post_init__(self):
        if (self.client_id < 0 or self.source_round < 0 or not self.class_counts or
                any(int(value) != value or value < 0 for value in self.class_counts) or
                sum(self.class_counts) <= 0):
            raise ValueError("invalid label histogram report")


@dataclass(frozen=True)
class LabelShiftReport:
    client_id: int
    source_round: int
    baseline_counts: tuple[int, ...]
    current_counts: tuple[int, ...]

    def __post_init__(self):
        vectors = (self.baseline_counts, self.current_counts)
        if (self.client_id < 0 or self.source_round < 0 or
                not self.baseline_counts or
                len(self.baseline_counts) != len(self.current_counts) or
                any(int(value) != value or value < 0
                    for counts in vectors for value in counts) or
                any(sum(counts) <= 0 for counts in vectors)):
            raise ValueError("invalid label shift report")


@dataclass(frozen=True)
class InputShiftReport:
    client_id: int
    source_round: int
    baseline_sum: float
    baseline_count: int
    current_sum: float
    current_count: int

    def __post_init__(self):
        if (self.client_id < 0 or self.source_round < 0 or
                self.baseline_count <= 0 or self.current_count <= 0 or
                any(not math.isfinite(float(value))
                    for value in (self.baseline_sum, self.current_sum))):
            raise ValueError("invalid input shift report")


@dataclass(frozen=True)
class FedDAAReport:
    client_id: int
    source_round: int
    sample_count: int
    n_classes: int
    prototype: tuple[float, ...]
    class_counts: tuple[int, ...]
    model_losses: tuple[tuple[int, float], ...]
    schema_version: int = 1

    def __post_init__(self):
        model_ids = [model_id for model_id, _ in self.model_losses]
        if (self.schema_version != 1 or self.client_id < 0 or self.source_round < 0 or
                self.sample_count <= 0 or
                self.n_classes <= 0 or len(self.prototype) != self.n_classes ** 2 or
                len(self.class_counts) != self.n_classes or
                sum(self.class_counts) != self.sample_count or not self.model_losses or
                len(model_ids) != len(set(model_ids))):
            raise ValueError("invalid FedDAA report identity")
        if (any(not math.isfinite(float(value)) for value in self.prototype) or
                any(int(value) != value or value < 0 for value in self.class_counts) or
                any(model_id < 0 or loss < 0 or not math.isfinite(float(loss))
                    for model_id, loss in self.model_losses)):
            raise ValueError("invalid FedDAA report payload")

    def losses(self):
        return {model_id: float(loss) for model_id, loss in self.model_losses}

    @property
    def communication_bytes(self):
        return 8 * (5 + len(self.prototype) + len(self.class_counts) +
                    2 * len(self.model_losses))


@dataclass
class ClientRuntimeState:
    availability: bool = True
    last_success_round: int | None = None
    selection_attempts: int = 0
    trusted_cache: UpdateEnvelope | None = None
    cache_age: int | None = None
    rejection_reason: str = ""
    return_state: str = "never_selected"
    unavailable_streak: int = 0
    cooldown_until: int = 0
    returned_round: int | None = None

    def observe_availability(self, available: bool, current_round: int,
                             selected: bool = False):
        was_available = self.availability
        self.availability = bool(available)
        if selected:
            self.selection_attempts += 1
        if self.availability and not was_available:
            self.return_state = "returned"
            self.returned_round = current_round
            self.unavailable_streak = 0
            self.cooldown_until = current_round
        elif not self.availability:
            self.return_state = "unavailable"
            self.unavailable_streak += 1
            self.cooldown_until = current_round + min(self.unavailable_streak, 5)

    def usable_cache(self, current_round: int, ttl: int):
        cached = self.trusted_cache
        if cached is None or cached.trust_state != "trusted":
            self.cache_age = None
            return None
        self.cache_age = current_round - cached.source_round
        if self.cache_age > ttl:
            self.trusted_cache = None
            self.rejection_reason = "cache_expired"
            return None
        return cached

    def accept(self, update: UpdateEnvelope):
        self.trusted_cache = update.trusted()
        self.last_success_round = update.source_round
        self.cache_age = 0
        self.rejection_reason = ""
        self.return_state = "trained"

    def reject(self, reason: str):
        self.trusted_cache = None
        self.cache_age = None
        self.rejection_reason = reason
