"""Decision interface + baseline agents (B1 + dropout).

Action space: no_op | drift_adapt | moment_align_adapt | label_prior_adapt | reweight | robust |
dropout_handle | friend_substitute | fedprox | spawn_concept_auto.
Three disturbances, three matched tools; the same fixed-tool agent wins in its
own scenario and loses in the others. Only an agent that DIAGNOSES wins all.

  drift   -> drift_adapt / moment_align_adapt / label_prior_adapt
  fault   -> robust          (reject; adapt/handle are wrong)
  dropout -> dropout_handle  (MIFA/FedVARP stale-update compensation)
"""
import json
import hashlib
import time
import numpy as np
from action_contract import (
    ACTION_SPECS, B3_VISIBLE_ACTIONS, Action, ActionBundle, as_action_bundle,
)
from observation_contract import ablate_observation, serialize_llm_observation

ACTION_FAMILY = {name: spec.family for name, spec in ACTION_SPECS.items()}


class Agent:
    name = "base"
    def decide(self, obs) -> Action | ActionBundle:
        raise NotImplementedError


class NoOpAgent(Agent):
    name = "noop"
    def decide(self, obs):
        return Action("no_op")


class RandomAgent(Agent):
    """Lower-bound baseline (AC5.4.1): samples a legal action at random, with a
    low intervention probability so it isn't a pure no_op spammer. Seeded for
    reproducibility (run.py reseeds globals per agent; this rng is independent)."""
    name = "random"
    ACTIONS = ["drift_adapt", "moment_align_adapt", "retain_history_adapt", "lr_reset", "robust", "dropout_handle", "friend_substitute",
               "fedprox", "spawn_concept", "spawn_concept_auto", "label_prior_adapt", "reweight"]
    def __init__(self, p_act=0.1, seed=0, max_actions=1, actions=None):
        self.p_act = p_act
        self.rng = np.random.default_rng(seed)
        self.max_actions = max(1, min(3, int(max_actions)))
        self.actions = list(actions or self.ACTIONS)
    def decide(self, obs):
        if self.rng.random() < self.p_act:
            if self.max_actions == 1:
                return Action(str(self.rng.choice(self.actions)))
            picked, families = [], set()
            for name in self.rng.permutation(self.actions):
                name = str(name)
                family = ACTION_FAMILY[name]
                if family not in families:
                    picked.append(name); families.add(family)
                if len(picked) == self.max_actions:
                    break
            return ActionBundle(tuple(picked))
        return Action("no_op")


class RandomBundleV2Agent(RandomAgent):
    """Random lower bound with a uniform 1--3 action-bundle size."""
    name = "random_bundle_v2"

    def __init__(self, p_act=0.1, seed=0, max_actions=3, actions=None):
        super().__init__(p_act=p_act, seed=seed, max_actions=max_actions, actions=actions)
        unknown = set(self.actions) - set(ACTION_FAMILY)
        if unknown:
            raise ValueError(f"unknown random action(s): {sorted(unknown)}")
        self.actions = [name for name in self.actions if name != "no_op"]
        if not self.actions:
            raise ValueError("random_bundle_v2 requires at least one active action")

    def decide(self, obs):
        if self.rng.random() >= self.p_act:
            return Action("no_op")
        by_family = {}
        for name in self.actions:
            by_family.setdefault(ACTION_FAMILY[name], []).append(name)
        size = int(self.rng.integers(1, min(self.max_actions, len(by_family)) + 1))
        families = self.rng.choice(tuple(by_family), size=size, replace=False)
        actions = tuple(str(self.rng.choice(by_family[str(family)])) for family in families)
        return ActionBundle(actions)


class AdaptAgent(Agent):
    """Real-drift baseline: follows the new concept with drift_adapt."""
    name = "adapt_lr"
    def __init__(self, drop_thresh=-0.02):
        self.drop_thresh = drop_thresh
    def decide(self, obs):
        if obs["acc_delta_3"] < self.drop_thresh:
            return Action("drift_adapt")
        return Action("no_op")


class InputShiftAgent(Agent):
    """Single-tool specialist for observable virtual/input shift."""
    name = "moment_align"
    def __init__(self, in_thresh=1.0):
        self.in_thresh = in_thresh
    def decide(self, obs):
        if obs.get("input_shift", 0.0) > self.in_thresh:
            return Action("moment_align_adapt")
        return Action("no_op")


class RejectAgent(Agent):
    """Confirm relative norm spikes and keep robust aggregation until recovery."""
    name = "reject_robust"
    def __init__(self, ratio_thresh=1.6, release_thresh=1.3, vote_window=4):
        self.ratio_thresh = ratio_thresh
        self.release_thresh = release_thresh
        self.vote_window = vote_window
        self.votes = []
        self.active = False
        self.release_votes = 0
    def decide(self, obs):
        ratio = obs.get("update_norm_ratio", obs.get("update_norm_outlier", 1.0))
        self.votes = (self.votes + [ratio >= self.ratio_thresh])[-self.vote_window:]
        if not self.active and sum(self.votes) >= 2:
            self.active = True
        if self.active:
            self.release_votes = self.release_votes + 1 if ratio < self.release_thresh else 0
            if self.release_votes >= 2:
                self.active = False
                self.votes = []
        return Action("robust") if self.active else Action("no_op")


class HandleDropoutAgent(Agent):
    """MIFA/FedVARP-style stale-update compensation for missing clients."""
    name = "handle_dropout"
    def decide(self, obs):
        gap = obs.get("participation_gap")
        if (gap is not None and gap > 0.0) or (
                gap is None and obs["participation_rate"] < 0.999):
            return Action("dropout_handle")
        return Action("no_op")


class FriendSubstituteAgent(Agent):
    """FL-FDMS-style dropout response: substitute absent clients with online friends."""
    name = "friend_substitute"
    def decide(self, obs):
        gap = obs.get("participation_gap")
        if (gap is not None and gap > 0.0) or (
                gap is None and obs["participation_rate"] < 0.999):
            return Action("friend_substitute")
        return Action("no_op")


class SwitchProxAgent(Agent):
    """Supported FedProx ablation; formally wrong for the current B1 workpoint."""
    name = "switch_prox"
    def __init__(self, var_thresh=0.006, window=3):
        self.var_thresh = var_thresh
        self.window = window
        self.vars = []
    def decide(self, obs):
        self.vars.append(obs["update_norm_var"])
        recent = self.vars[-self.window:]
        persistent = len(recent) == self.window and float(np.mean(recent)) > self.var_thresh
        no_cliff = obs["acc_delta_3"] > -0.03
        if persistent and no_cliff:
            return Action("fedprox")
        return Action("no_op")


class SpawnConceptAgent(Agent):
    """Right for staggered drift (clients drift to DIFFERENT concepts -> a single
    model is torn between them; only per-concept models fix it). step1 fires on an
    accuracy dip -> spawn_concept (engine uses ground-truth clustering = FedDrift
    oracle bound); automatic cluster diagnosis is step2. In non-staggered scenarios
    the engine ignores spawn_concept, so this agent acts as a (wrong-tool) probe."""
    name = "spawn_concept"
    def __init__(self, drop_thresh=-0.02):
        self.drop_thresh = drop_thresh
    def decide(self, obs):
        # fires on the FIRST dip (G1 wave), not waiting for a second wave: spawning
        # early is harmless — a not-yet-drifted cluster just trains the original
        # concept until its own drift arrives. (plan said "second dip"; first is simpler.)
        if obs["acc_delta_3"] < self.drop_thresh:
            return Action("spawn_concept")
        return Action("no_op")


class AutoSpawnConceptAgent(Agent):
    """Staggered drift response without ground-truth clusters."""
    name = "spawn_concept_auto"
    def __init__(self, score_floor=0.18, relative_threshold=0.45,
                 min_fraction=0.10, max_fraction=0.75,
                 roster_threshold=0.60, cohort_threshold=0.50,
                 separation_threshold=0.05):
        self.score_floor = score_floor
        self.relative_threshold = relative_threshold
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.roster_threshold = roster_threshold
        self.cohort_threshold = cohort_threshold
        self.separation_threshold = separation_threshold
        self.candidate_votes = []
        self.direction_votes = []
        self.pending = False
        self.fired = False
        self.virtual_blocked = False
    def decide(self, obs):
        # ponytail: the current core track excludes virtual+staggered; revisit
        # this latch only when that deferred composite is added.
        self.virtual_blocked = (
            self.virtual_blocked or float(obs.get("input_shift", 0.0)) > 1.0
        )
        if self.virtual_blocked:
            self.pending = False
            return Action("no_op")
        p50 = float(obs.get("client_model_score_drop_p50", 0.0))
        p90 = float(obs.get("client_model_score_drop_p90", 0.0))
        tail_contrast = (p90 - p50) / max(p90, 1e-6)
        planned_partial = float(obs.get("planned_participation_rate", 1.0)) < 0.999
        # ponytail: core-track guard; replace with cause arbitration before
        # evaluating the deferred staggered+dropout combination.
        actual_dropout = float(obs.get("participation_gap", 0.0)) > 0.0
        candidate = (
            bool(obs.get("client_change_monitor_ready")) and
            not planned_partial and not actual_dropout and
            p90 >= self.score_floor and
            tail_contrast >= self.relative_threshold and
            self.min_fraction <= float(
                obs.get("client_score_exceedance_fraction", 0.0)
            ) <= self.max_fraction and
            float(obs.get("online_roster_overlap", 0.0)) >= self.roster_threshold
        )
        separation = float(obs.get("client_update_group_separation", 0.0))
        direction_candidate = candidate and separation >= self.separation_threshold
        self.pending = candidate
        self.candidate_votes = (self.candidate_votes + [candidate])[-3:]
        self.direction_votes = (self.direction_votes + [direction_candidate])[-3:]
        stable_group = (
            candidate and
            float(obs.get("client_score_exceedance_overlap", 0.0)) >=
            self.cohort_threshold
        )
        if (not self.fired and sum(self.candidate_votes[-3:]) >= 2 and
                any(self.direction_votes[-3:]) and stable_group):
            self.fired = True
            return Action("spawn_concept_auto")
        return Action("no_op")


class LabelPriorAdaptAgent(Agent):
    """Right for global label-prior drift: correct logits by target/source priors."""
    name = "label_prior_adapt"
    def __init__(self, lab_thresh=0.25):
        self.lab_thresh = lab_thresh
    def decide(self, obs):
        if obs.get("label_shift", 0) > self.lab_thresh:
            return Action("label_prior_adapt")
        return Action("no_op")


class ReweightAgent(Agent):
    """FedLC baseline for client label-skew; kept for A/B against prior correction."""
    name = "reweight"
    def __init__(self, lab_thresh=0.25):
        self.lab_thresh = lab_thresh
    def decide(self, obs):
        if obs.get("label_shift", 0) > self.lab_thresh:
            return Action("reweight")
        return Action("no_op")


class SelectClientsAgent(Agent):
    """Oort-style participant selection from utility, exploration, and system cost."""
    name = "select_clients"
    def _pick(self, rows, k, rng, key):
        if k <= 0 or not rows:
            return []
        weights = np.asarray([max(0.0, float(r.get(key, 0.0))) for r in rows], dtype=float)
        p = weights / weights.sum() if weights.sum() > 0 else None
        take = min(k, len(rows))
        idx = rng.choice(len(rows), take, replace=False, p=p)
        return [rows[int(i)] for i in np.atleast_1d(idx)]

    def decide(self, obs):
        utils = obs.get("client_utilities", [])
        k = int(obs.get("select_k", 0) or 0)
        if not obs.get("select_enabled") or k <= 0 or not utils:
            return Action("no_op")
        rng = np.random.default_rng(int(obs.get("select_seed", 0)))
        rows = [u for u in utils
                if not u.get("oort_blacklisted") and u.get("available", True)]
        explore_k = int(round(k * float(obs.get("oort_exploration", 0.0))))
        window = max(1, int(obs.get("oort_sample_window", 5)))
        explore = sorted([u for u in rows if u.get("oort_unexplored")],
                         key=lambda u: u.get("oort_reward", 0.0), reverse=True)[:max(explore_k, 1) * window]
        exploit = sorted([u for u in rows if not u.get("oort_unexplored")],
                         key=lambda u: u.get("oort_score", u.get("utility", 0.0)), reverse=True)[:max(k - explore_k, 1) * window]
        chosen = self._pick(exploit, k - explore_k, rng, "oort_score") + self._pick(explore, explore_k, rng, "oort_reward")
        seen = {int(u["client"]) for u in chosen}
        for u in sorted(rows, key=lambda x: x.get("oort_score", x.get("utility", 0.0)), reverse=True):
            if len(chosen) >= k:
                break
            if int(u["client"]) not in seen:
                chosen.append(u); seen.add(int(u["client"]))
        return Action("select_clients", tuple(int(u["client"]) for u in chosen[:k]))


class DiagnoseAgent(Agent):
    """Rule baseline for drift sub-type diagnosis and matching B2 action."""
    name = "diagnose"
    def __init__(self, in_thresh=1.0, lab_thresh=0.25, drop_thresh=-0.08):
        self.in_thresh = in_thresh
        self.lab_thresh = lab_thresh
        self.drop_thresh = drop_thresh
        self.last_diagnosis = ""
        self.hetero_vars = []
        self.selector = SelectClientsAgent()
        self.hetero_detected = False
        self.real_votes = []
        self.label_votes = []
    def decide(self, obs):
        if self.hetero_detected:
            return self.selector.decide(obs)
        startup_window = max(4, 2 * int(obs.get("decision_interval", 1)))
        startup_partial = (obs.get("select_enabled") and obs.get("round", 0) <= startup_window and
                           obs.get("participation_rate", 1.0) < 0.999)
        self.hetero_vars = (self.hetero_vars + [obs.get("update_norm_var", 0.0)]) if startup_partial else []
        if len(self.hetero_vars) >= 3 and float(np.mean(self.hetero_vars[-3:])) > 0.006:
            self.hetero_detected = True
            return self.selector.decide(obs)
        if self.last_diagnosis == "real":
            return Action("drift_adapt")
        if self.last_diagnosis == "label":
            return Action("label_prior_adapt")
        if obs.get("input_shift", 0) > self.in_thresh:
            self.last_diagnosis = "virtual"
            return Action("moment_align_adapt")
        drop_thresh = (-0.03 if float(
            obs.get("planned_participation_rate", 1.0)
        ) < 0.999 else self.drop_thresh)
        cliff = obs.get("acc_delta_3", 0.0) < drop_thresh
        self.real_votes = (self.real_votes + [cliff])[-3:]
        label_vote = obs.get("label_shift", 0) >= self.lab_thresh and not cliff
        self.label_votes = (self.label_votes + [label_vote])[-3:]
        if sum(self.real_votes) >= 2:
            self.last_diagnosis = "real"
            return Action("drift_adapt")
        if sum(self.label_votes) >= 2:
            self.last_diagnosis = "label"
            return Action("label_prior_adapt")
        return Action("no_op")


class CompositeDiagnoseAgent(Agent):
    """B3 rule baseline: combine independent public symptoms into one bundle."""
    name = "composite_rule"

    def __init__(self, in_thresh=1.0, lab_thresh=0.25, drop_thresh=-0.08,
                 real_lab_thresh=0.05, real_fraction_thresh=0.50,
                 core_only=False, ablation_group=None):
        self.in_thresh = in_thresh
        self.lab_thresh = lab_thresh
        self.drop_thresh = drop_thresh
        self.real_lab_thresh = real_lab_thresh
        self.real_fraction_thresh = real_fraction_thresh
        self.core_only = core_only
        self.reject = RejectAgent()
        self.spawner = AutoSpawnConceptAgent()
        self.selector = SelectClientsAgent()
        self.hetero_vars = []
        self.hetero_detected = False
        self.real_votes = []
        self.label_votes = []
        self.data_diagnosis = ""
        self.staggered_detected = False
        self.availability_baseline = None
        if ablation_group is not None:
            self.ablation_group = ablation_group

    def decide(self, obs):
        obs = ablate_observation(obs, getattr(self, "ablation_group", None))
        actions = []
        startup_window = max(4, 2 * int(obs.get("decision_interval", 1)))
        startup_partial = (not self.core_only and
            obs.get("select_enabled") and
            obs.get("round", 0) <= startup_window and
            obs.get("participation_rate", 1.0) < 0.999
        )
        self.hetero_vars = (
            self.hetero_vars + [obs.get("update_norm_var", 0.0)]
            if startup_partial else []
        )
        if (len(self.hetero_vars) >= 3 and
                float(np.mean(self.hetero_vars[-3:])) > 0.006):
            self.hetero_detected = True

        robust = (Action("no_op") if startup_partial or self.hetero_detected or
                  self.staggered_detected
                  else self.reject.decide(obs))
        spawned = (Action("no_op") if self.core_only else self.spawner.decide(obs))
        if spawned.type != "no_op":
            self.staggered_detected = True
            # The same transient norm spike can precede a concept split. Restart
            # fault confirmation; a real coexisting fault will reconfirm later.
            self.reject = RejectAgent()
            robust = Action("no_op")

        drop_thresh = (-0.03 if float(
            obs.get("planned_participation_rate", 1.0)
        ) < 0.999 else self.drop_thresh)
        cliff = obs.get("acc_delta_3", 0.0) < drop_thresh
        real_vote = (
            cliff and
            obs.get("label_shift", 0.0) >= self.real_lab_thresh and
            obs.get("client_score_exceedance_fraction", 0.0) >=
            self.real_fraction_thresh and
            obs.get("input_shift", 0.0) <= self.in_thresh and
            not self.spawner.pending and
            not self.staggered_detected
        )
        self.real_votes = (self.real_votes + [real_vote])[-3:]
        label_vote = (
            not self.core_only and
            obs.get("label_shift", 0.0) >= self.lab_thresh and
            obs.get("input_shift", 0.0) <= self.in_thresh and
            not cliff and not self.staggered_detected
        )
        self.label_votes = (self.label_votes + [label_vote])[-3:]
        if obs.get("input_shift", 0.0) > self.in_thresh:
            self.data_diagnosis = "virtual"
        elif not self.data_diagnosis:
            if sum(self.real_votes) >= 2:
                self.data_diagnosis = "real"
            elif sum(self.label_votes) >= 2:
                self.data_diagnosis = "label"
        data_actions = {
            "virtual": "moment_align_adapt",
            "real": "drift_adapt",
            "label": "label_prior_adapt",
        }
        if self.data_diagnosis:
            actions.append(Action(data_actions[self.data_diagnosis]))
        if robust.type != "no_op":
            actions.append(robust)

        gap = float(obs.get("participation_gap", 0.0))
        availability = float(obs.get(
            "availability_rate", obs.get("participation_rate", 1.0)
        ))
        self.availability_baseline = (
            availability if self.availability_baseline is None
            else max(self.availability_baseline, availability)
        )
        if gap > 0.0 or availability < self.availability_baseline - 1e-6:
            actions.append(Action("dropout_handle"))

        if not self.core_only and self.hetero_detected:
            selected = self.selector.decide(obs)
            if selected.type != "no_op":
                actions.append(selected)
        if not self.core_only and spawned.type != "no_op":
            actions.append(spawned)
        return ActionBundle(tuple(actions[:3]))


class OracleAgent(Agent):
    """Upper bound: knows the scenario, applies the matched tool at drift_round.
    Also reports the ground-truth drift sub-type as its diagnosis (upper bound)."""
    name = "oracle"
    always_decide = True
    def __init__(self, scenario, drift_round, drift_type="real"):
        self.scenario = scenario
        self.drift_round = drift_round
        self.last_diagnosis = drift_type if scenario == "drift" else ""
    def decide(self, obs):
        if obs["round"] < self.drift_round:
            return Action("no_op")
        if self.scenario == "drift":
            if self.last_diagnosis == "label":
                return Action("label_prior_adapt")
            if self.last_diagnosis == "virtual":
                return Action("moment_align_adapt")
            return Action("drift_adapt")
        if self.scenario == "fault":
            return Action("robust") if not obs["using_robust"] else Action("no_op")
        if self.scenario == "dropout":
            return Action("dropout_handle")
        if self.scenario == "hetero":
            return SelectClientsAgent().decide(obs)
        if self.scenario == "staggered":
            # spawn from G1's drift round (drift_round), not G2's: early clustering
            # is harmless and >= waiting. (plan said G2's round; G1 is simpler.)
            return Action("spawn_concept")
        return Action("no_op")


class CompositeOracleAgent(Agent):
    """Hidden-truth upper bound for a frozen B3 canonical bundle."""
    name = "composite_oracle"
    always_decide = True

    def __init__(self, event_round, canonical_bundle):
        self.event_round = event_round
        self.template = ActionBundle(tuple(canonical_bundle))
        self.selector = SelectClientsAgent()
        self.completed_one_shot = set()

    def decide(self, obs):
        if obs["round"] < self.event_round:
            return Action("no_op")
        actions = []
        for action in self.template.actions:
            if (ACTION_SPECS[action.type].lifecycle == "one_shot_irreversible" and
                    action.type in self.completed_one_shot):
                continue
            selected = self.selector.decide(obs) if action.type == "select_clients" else action
            if selected.type != "no_op":
                actions.append(selected)
                if ACTION_SPECS[selected.type].lifecycle == "one_shot_irreversible":
                    self.completed_one_shot.add(selected.type)
        return ActionBundle(tuple(actions))


class LLMAgent(Agent):
    """Main subject (AC5.5.1). Default offline stub does NOT yet diagnose (it
    assumes drift -> lr_reset on any drop), which is exactly why it loses in the
    fault/dropout scenarios. The real LLM should read the symptoms and pick
    lr_reset vs robust vs dropout_handle itself. Replace offline_query_fn."""
    name = "llm"
    ACTION_PRIORITY = (
        "spawn_concept_auto", "spawn_concept", "select_clients", "moment_align_adapt",
        "retain_history_adapt", "label_prior_adapt", "reweight", "fedprox",
        "friend_substitute", "dropout_handle", "robust", "drift_adapt", "lr_reset", "no_op",
    )
    MEMORY_METRICS = (
        "global_acc", "client_loss_mean", "update_norm_ratio", "participation_gap",
        "client_model_score_drop_p50", "client_model_score_drop_p90",
        "client_update_direction_dispersion",
        "online_roster_overlap", "client_score_exceedance_fraction",
        "client_score_exceedance_overlap", "client_update_group_separation",
    )

    def __init__(self, query_fn=None, allowed_actions=None, max_calls=None, ablation_group=None):
        self.query_fn = query_fn or offline_query_fn
        self.allowed_actions = tuple(allowed_actions or self.ACTION_PRIORITY)
        self.last_reasoning = ""
        self.last_diagnosis = ""
        self.selector = SelectClientsAgent()
        self.called_this_round = False
        self.last_latency_ms = 0.0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_cost_usd = 0.0
        self.last_api_attempts = 0
        self.last_request_sha256 = ""
        self.last_public_observation_json = ""
        self.last_replayed = False
        self.last_parse_error = ""
        self.last_api_error = ""
        self.last_budget_error = ""
        self.last_raw_output = ""
        self.last_declared_actions = ()
        self.last_bundle_status = "VALID"
        self.max_calls = None if max_calls is None else int(max_calls)
        if ablation_group is not None:
            self.ablation_group = ablation_group
        self.calls_made = 0
        self.model_name = getattr(self.query_fn, "model_name", "offline")

        self._last_decision = None
        self._last_decision_observation = None
        self._action_start_rounds = {}

    def _decision_memory(self, obs):
        previous = self._last_decision_observation
        deltas = ({
            key: round(float(obs[key]) - float(previous[key]), 6)
            for key in self.MEMORY_METRICS
        } if previous is not None else {})
        current_round = int(obs["round"])
        ages = {
            action: max(1, current_round - self._action_start_rounds.get(action, current_round) + 1)
            for action in obs.get("active_actions", ())
        }
        return {
            "last_decision": self._last_decision,
            "active_action_ages": ages,
            "telemetry_delta_since_last_decision": deltas,
        }

    def record_decision_feedback(self, *, round_idx, declared_actions, status,
                                 activated_actions, started_actions, stopped_actions,
                                 effective_from_round, observation):
        for action in stopped_actions:
            self._action_start_rounds.pop(action, None)
        for action in started_actions:
            self._action_start_rounds[action] = int(effective_from_round)
        self._last_decision = {
            "round": int(round_idx),
            "declared_actions": list(declared_actions),
            "status": str(status),
            "activated_actions": list(activated_actions) if status == "VALID" else [],
        }
        self._last_decision_observation = {
            key: float(observation[key]) for key in self.MEMORY_METRICS
        }

    def decide(self, obs):
        self.last_parse_error = ""
        self.last_api_error = ""
        self.last_budget_error = ""
        self.last_raw_output = ""
        self.last_declared_actions = ()
        self.last_bundle_status = "VALID"
        self.last_api_attempts = 0
        self.last_request_sha256 = ""
        self.last_replayed = False
        public_observation = serialize_llm_observation(
            obs, self._decision_memory(obs), getattr(self, "ablation_group", None)
        )
        self.last_public_observation_json = public_observation
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            self.last_budget_error = "budget_exhausted"
            return Action("no_op")
        self.calls_made += 1
        self.called_this_round = True
        self.last_request_sha256 = hashlib.sha256(
            public_observation.encode("utf-8")
        ).hexdigest()
        started = time.perf_counter()
        try:
            raw = self.query_fn(public_observation)
        except Exception as e:
            self.last_reasoning = f"agent error: {e}"
            self.last_api_error = type(e).__name__
            self.last_api_attempts = int(getattr(self.query_fn, "last_attempts", 1) or 1)
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            return Action("no_op")
        measured_latency = (time.perf_counter() - started) * 1000
        self.last_latency_ms = float(
            getattr(self.query_fn, "last_latency_ms", measured_latency)
        )
        usage = getattr(self.query_fn, "last_usage", {})
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_cost_usd = float(getattr(self.query_fn, "last_cost_usd", 0.0) or 0.0)
        self.last_api_error = str(getattr(self.query_fn, "last_error", "") or "")
        self.last_api_attempts = int(getattr(self.query_fn, "last_attempts", 1) or 1)
        self.last_request_sha256 = str(
            getattr(self.query_fn, "last_request_sha256", self.last_request_sha256)
        )
        self.last_replayed = bool(getattr(self.query_fn, "last_replayed", False))
        if not isinstance(raw, str):
            self.last_raw_output = repr(raw)
            self.last_parse_error = "response is not text"
            self.last_bundle_status = "INVALID_SCHEMA"
            return ActionBundle(("no_op",))
        self.last_raw_output = raw
        self.last_reasoning = raw.strip().replace("\n", " ")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not set(payload) <= {"actions", "subtype"}:
                raise ValueError("response must be a JSON object with only actions/subtype")
            names = payload.get("actions")
            if (not isinstance(names, list) or not 1 <= len(names) <= 3 or
                    any(type(name) is not str for name in names)):
                raise ValueError("actions must contain one to three strings")
            subtype = payload.get("subtype")
            if subtype is not None and subtype not in {"real", "virtual", "label"}:
                raise ValueError("invalid subtype")
            names = [name.lower() for name in names]
            if any(name not in self.allowed_actions for name in names):
                self.last_bundle_status = "UNKNOWN_ACTION"
                raise ValueError("action is not allowed")
            if len(names) != len(set(names)):
                self.last_bundle_status = "CONFLICT"
                raise ValueError("duplicate actions are not allowed")
            self.last_diagnosis = subtype or ""
        except (json.JSONDecodeError, ValueError) as exc:
            self.last_parse_error = str(exc)
            if self.last_bundle_status == "VALID":
                self.last_bundle_status = "INVALID_SCHEMA"
            return ActionBundle(("no_op",))
        actions = []
        for name in names:
            if name == "select_clients":
                selected = self.selector.decide(obs)
                if selected.type != "no_op":
                    actions.append(selected)
            else:
                actions.append(Action(name))
        try:
            bundle = ActionBundle(tuple(actions))
        except (TypeError, ValueError) as exc:
            self.last_parse_error = str(exc)
            self.last_bundle_status = "CONFLICT"
            return ActionBundle(("no_op",))
        self.last_declared_actions = tuple(names)
        return bundle


def offline_query_fn(text):
    """PLACEHOLDER. Send `text` as the user message with a system prompt that
    describes the action space (no_op / lr_reset / robust / friend_substitute) and
    asks the model to diagnose drift vs fault vs dropout from the symptoms."""
    obs = json.loads(text)
    if obs["aggregate_label_hist_tv"] > 0.25:
        return json.dumps({"actions": ["label_prior_adapt"], "subtype": "label"})
    if obs["input_moment_delta"] > 1.0:
        return json.dumps({"actions": ["moment_align_adapt"], "subtype": "virtual"})
    if (obs["round"] <= 4 and obs["update_norm_var"] > 0.006 and
            obs["participation_rate"] < 0.999):
        return json.dumps({"actions": ["select_clients"]})
    if obs["acc_delta_3"] < -0.08:
        return json.dumps({"actions": ["drift_adapt"], "subtype": "real"})
    return json.dumps({"actions": ["no_op"]})


def make_agents(scenario, seed, event_round, llm_query_fn=None, drift_type="real"):
    """Full benchmark roster — single source of truth for run.py / multi_seed.py.
    llm_query_fn: real-LLM backend for LLMAgent (None -> offline stub)."""
    return [NoOpAgent(), RandomAgent(seed=seed), AdaptAgent(), RejectAgent(),
            HandleDropoutAgent(), FriendSubstituteAgent(), SwitchProxAgent(),
            SpawnConceptAgent(), AutoSpawnConceptAgent(), LabelPriorAdaptAgent(), ReweightAgent(),
            SelectClientsAgent(), DiagnoseAgent(),
            OracleAgent(scenario, event_round, drift_type), LLMAgent(query_fn=llm_query_fn)]
