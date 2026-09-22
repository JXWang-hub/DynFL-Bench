"""Shared action schema and lifecycle reducer; stdlib-only by design."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    family: str
    stage: int
    lifecycle: str


ACTION_SPECS = {
    "no_op": ActionSpec("control", -1, "control"),
    "select_clients": ActionSpec("participant_selection", 0, "per_round"),
    "dropout_handle": ActionSpec("missing_client_handling", 1, "reversible_state"),
    "friend_substitute": ActionSpec("missing_client_handling", 1, "reversible_state"),
    "drift_adapt": ActionSpec("data_adaptation", 2, "reversible_state"),
    "moment_align_adapt": ActionSpec("data_adaptation", 2, "reversible_state"),
    "label_prior_adapt": ActionSpec("data_adaptation", 2, "reversible_state"),
    "lr_reset": ActionSpec("data_adaptation", 2, "per_round"),
    "retain_history_adapt": ActionSpec("data_adaptation", 2, "reversible_state"),
    "reweight": ActionSpec("data_adaptation", 2, "reversible_state"),
    "fedprox": ActionSpec("local_optimization", 2, "reversible_state"),
    "robust": ActionSpec("aggregation_defense", 3, "reversible_state"),
    "spawn_concept": ActionSpec("concept_routing", 4, "one_shot_irreversible"),
    "spawn_concept_auto": ActionSpec("concept_routing", 4, "one_shot_irreversible"),
}

B3_VISIBLE_ACTIONS = (
    "no_op", "drift_adapt", "moment_align_adapt", "label_prior_adapt", "robust",
    "dropout_handle", "friend_substitute", "select_clients", "spawn_concept_auto",
)


@dataclass(frozen=True)
class Action:
    type: str
    clients: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActionBundle:
    actions: tuple[Action | str, ...]

    def __post_init__(self):
        normalized = []
        for raw in self.actions:
            action = raw if isinstance(raw, Action) else Action(raw) if isinstance(raw, str) else None
            if action is None:
                raise TypeError("ActionBundle entries must be Action or str")
            if action.type not in ACTION_SPECS:
                raise ValueError(f"unknown action: {action.type}")
            if action.type not in {item.type for item in normalized}:
                normalized.append(action)
        if not normalized:
            normalized = [Action("no_op")]
        if len(normalized) > 3:
            raise ValueError("ActionBundle accepts at most 3 distinct actions")
        if len(normalized) > 1 and any(action.type == "no_op" for action in normalized):
            raise ValueError("no_op cannot coexist with active actions")
        families = [ACTION_SPECS[action.type].family for action in normalized]
        if len(families) != len(set(families)):
            raise ValueError("only one action per family is allowed")
        object.__setattr__(self, "actions", tuple(normalized))

    @property
    def action_types(self):
        return tuple(action.type for action in self.actions)

    @property
    def ordered_actions(self):
        return tuple(sorted(self.actions, key=lambda action: ACTION_SPECS[action.type].stage))

    @property
    def type(self):
        return self.actions[0].type if len(self.actions) == 1 else "bundle"

    @property
    def clients(self):
        return self.actions[0].clients if len(self.actions) == 1 else ()


def as_action_bundle(value: Action | ActionBundle) -> ActionBundle:
    if isinstance(value, ActionBundle):
        return value
    if isinstance(value, Action):
        return ActionBundle((value,))
    raise TypeError("Agent.decide() must return Action or ActionBundle")


@dataclass(frozen=True)
class ActionState:
    reversible: tuple[str, ...] = ()
    irreversible: tuple[str, ...] = ()
    per_round: tuple[str, ...] = ()

    @property
    def active(self) -> tuple[str, ...]:
        names = (*self.reversible, *self.irreversible, *self.per_round)
        return tuple(sorted(dict.fromkeys(names), key=lambda name: ACTION_SPECS[name].stage))


@dataclass(frozen=True)
class ActionTransition:
    state: ActionState
    emitted: tuple[str, ...]
    active_before: tuple[str, ...]
    active_after: tuple[str, ...]
    started: tuple[str, ...]
    stopped: tuple[str, ...]
    one_shot: tuple[str, ...]


def reduce_action_state(previous: ActionState, emitted: ActionBundle | None) -> ActionTransition:
    """Apply one decision; ``None`` means no decision was due this round."""
    before = previous.active
    if emitted is None:
        state = ActionState(previous.reversible, previous.irreversible)
        return ActionTransition(state, (), before, state.active, (), previous.per_round, ())

    names = emitted.action_types
    if names == ("no_op",):
        state = ActionState((), previous.irreversible)
        return ActionTransition(
            state, names, before, state.active, (),
            tuple((*previous.reversible, *previous.per_round)), (),
        )

    reversible = tuple(name for name in names
                       if ACTION_SPECS[name].lifecycle == "reversible_state")
    per_round = tuple(name for name in names
                      if ACTION_SPECS[name].lifecycle == "per_round")
    one_shot = tuple(name for name in names
                     if ACTION_SPECS[name].lifecycle == "one_shot_irreversible"
                     and name not in previous.irreversible)
    irreversible = tuple(dict.fromkeys((*previous.irreversible, *one_shot)))
    state = ActionState(reversible, irreversible, per_round)
    after = state.active
    return ActionTransition(
        state, names, before, after,
        tuple(name for name in after if name not in before),
        tuple(name for name in before if name not in after),
        one_shot,
    )
