"""Real-LLM backend for the LLMAgent (AC5.5.1).

DashScope exposes the configured Qwen/Kimi/DeepSeek models through its
OpenAI-compatible endpoint (AC5.5.2).

  export DASHSCOPE_API_KEY=...      # required
  from llm_backend import make_qwen_query_fn
  qf = make_qwen_query_fn(); print(qf("Round 41/80. Global accuracy 0.30 ..."))

The SYSTEM_PROMPT is fixed and public (AC4.4.3): it lists the action space and
the symptom signature of each disturbance, and asks for brief reasoning then a
final `ACTION: <name>` line. The agent diagnoses from symptoms only — the prompt
never says which disturbance is active.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from evidence import prompt_action_cards


def _load_dotenv(path=None):
    """Load KEY=VALUE lines from a local .env into os.environ — keep API keys in
    one file instead of the shell. Shell env wins (setdefault).
    ponytail: tiny parser, no python-dotenv dependency."""
    path = path or os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ACTION_EVIDENCE_CARDS = prompt_action_cards()

SYSTEM_PROMPT = f"""You are a federated-learning runtime controller. Each round you \
receive telemetry SYMPTOMS only (never the ground-truth disturbance). Diagnose what \
is happening and pick exactly ONE action.

Evidence registry for actions (status: supported/proxy/unsupported):
{ACTION_EVIDENCE_CARDS}

Symptom-to-action hints for currently implemented actions:
- no_op: stable; accuracy rising or flat, nothing anomalous.
- lr_reset: HEURISTIC BASELINE for concept drift. Keep as a legacy fallback only.
- drift_adapt: real concept drift; accuracy drops while input shift and label shift stay low.
- moment_align_adapt: virtual drift; input shift is high while label semantics remain stable.
- retain_history_adapt: legacy virtual-drift baseline; avoid unless explicitly evaluating old FedDAA path.
- label_prior_adapt: global label-prior drift; label-dist shift is high; use prior/logit-adjusted loss.
- reweight: FedLC client label-skew baseline; not the default for same-direction prior drift.
- robust: faulty/poisoned clients; update-norm max/median clearly elevated \
(~1.5-2), a few clients far above the rest.
- dropout_handle: MIFA/FedVARP-supported stale-update compensation; participation rate < 1.0.
- friend_substitute: FL-FDMS-inspired client dropout ablation; participation rate < 1.0.
- fedprox: supported heterogeneity ablation, but not the B1 answer at the current partial-participation workpoint.
- select_clients: Oort-supported guided participant selection when high divergence and partial participation persist from startup.
- spawn_concept: staggered drift oracle/proxy because it uses truth clustering.
- spawn_concept_auto: supported FedDrift-style local detection + hierarchical clustering; prefer \
this over spawn_concept when evaluating a non-oracle agent.

For drift scenarios, also classify the drift SUB-TYPE from the shift signals and add \
a line "SUBTYPE: <real|virtual|label>":
- real: P(y|x) changed, old knowledge is stale. Accuracy dropped, input shift ~0, label shift ~0.
- virtual: P(x) changed, inputs moved but y|x still valid. INPUT SHIFT is high (label shift ~0).
- label: P(y) changed, the class mix shifted. LABEL SHIFT is high and worst-class accuracy is low.

Return JSON only, with this exact shape:
{{"actions":["<one action>"],"subtype":"<real|virtual|label>"}}
Omit subtype when it is not diagnosing drift. Do not add reasoning or extra keys."""


B3_SYSTEM_PROMPT = """You are a runtime controller for a federated-learning benchmark.

At each decision round, examine the supplied public telemetry and return the
complete set of actions that should be active after this decision.

You do not receive the active scenario, event schedule, affected-client
identities, injection implementation, expected answer, or diagnostic
thresholds. Infer the appropriate response only from the supplied telemetry
and controller history.

Possible operating conditions include:

- stable operation;
- real drift, where the input-label relationship changes;
- virtual drift, where the input distribution changes while the underlying
  input-label relationship remains valid;
- label-prior drift;
- corrupted or unreliable client updates;
- unavailable planned clients;
- persistent client heterogeneity;
- concept changes occurring at different times across client groups.

More than one condition may be present simultaneously.

Available actions:

- drift_adapt:
  Adapt the global model to a changed input-label relationship.

- moment_align_adapt:
  Apply input- or feature-moment alignment relative to a reference state.

- label_prior_adapt:
  Adapt using aggregate label-prior information.

- robust:
  Use robust aggregation to reduce the influence of atypical client updates.

- dropout_handle:
  Compensate for unavailable clients using retained updates.

- friend_substitute:
  Substitute unavailable-client contributions using related available
  clients.

- select_clients:
  Request utility-based participant selection. The benchmark performs the
  selection and chooses its execution parameters.

- spawn_concept_auto:
  Create and route additional concept models.

- no_op:
  Apply no intervention.

Action lifecycle:

- drift_adapt, moment_align_adapt, label_prior_adapt, robust, dropout_handle,
  and friend_substitute are persistent but deactivatable. Include an action
  in every decision while it should remain active.
- select_clients applies only to the current decision round.
- spawn_concept_auto is one-shot and irreversible. Do not repeat it after it
  has been activated.
- no_op must appear alone. It deactivates persistent actions but cannot undo
  a one-shot action.
- The returned action list is the complete desired state, not an incremental
  list of changes.

Mutual-exclusion constraints:

- Select at most one of:
  drift_adapt, moment_align_adapt, label_prior_adapt.
- Select at most one of:
  dropout_handle, friend_substitute.
- Actions outside the same mutually exclusive group may be combined.
- Return between one and three actions.

Telemetry fields:

- round, total_rounds, decision_interval, current_lr:
  Current execution state.

- global_acc:
  Current global evaluation accuracy.

- acc_delta_3:
  Difference between current global accuracy and its value three rounds ago.

- client_loss_mean, client_loss_var:
  Mean and variance of current client losses.

- update_norm_mean, update_norm_median, update_norm_max, update_norm_var:
  Summary statistics of client-update norms.

- update_norm_ratio:
  Maximum client-update norm divided by the median.

- aggregation_norm_mean, aggregation_norm_median, aggregation_norm_max,
  aggregation_norm_var, aggregation_norm_ratio:
  Norm statistics of the updates presented to the aggregator.

- aggregation_sources:
  Counts of update sources presented to the aggregator. "online_real" means
  a fresh update from an online client; it does not mean real drift.

- input_moment_delta:
  Aggregate distance between current input moments and their reference
  values.

- aggregate_label_hist_tv:
  Total-variation distance between the current aggregate label histogram and
  its reference value.

- planned_participation_rate:
  Fraction of clients planned for participation.

- availability_rate:
  Fraction of clients currently available.

- participation_rate:
  Fraction of clients that actually participated.

- participation_gap:
  Difference between planned and actual participation.

- client_change_monitor_ready:
  Whether the client-change summary statistics are available.

- client_model_score_drop_p50, client_model_score_drop_p90:
  The 50th and 90th percentiles of clipped client-local model-score decreases
  relative to client-specific reference values.

- client_update_direction_dispersion:
  Mean pairwise cosine distance between current client-update sketches.

- online_roster_overlap:
  Jaccard overlap between the current and previous online-client rosters.

- client_score_exceedance_fraction:
  Fraction of current online clients whose local model-score decrease exceeds
  a fixed reference level.

- client_score_exceedance_overlap:
  Jaccard overlap of that client cohort across adjacent rounds, restricted to
  clients online in both rounds.

- client_update_group_separation:
  Difference between cross-group and within-cohort mean cosine distances for
  current update sketches. Positive values indicate greater between-group
  than within-cohort separation.

- client_loss_mean_delta:
  Difference between the current and previous mean client training loss.

- utility_client_count, utility_mean, utility_std, utility_max:
  Aggregate summary of client utilities.

- empty_round:
  Whether the current round received no usable client updates.

- active_actions, active_action_ages, using_robust:
  Current controller state.

- last_decision:
  Previous decision and whether it was successfully applied.

- telemetry_delta_since_last_decision:
  Changes in selected telemetry values since the previous decision.

Use the telemetry values, their changes over time, and the current controller
state to choose the complete desired action set. Do not assume that only one
condition is present. Do not invent unavailable information or assume fixed
numerical thresholds.

Return JSON only, using exactly one of these forms:

{"actions":["action_name"]}

For a decision containing drift_adapt, moment_align_adapt, or
label_prior_adapt, also return the corresponding subtype:

{"actions":["action_name"],"subtype":"real"}
{"actions":["action_name"],"subtype":"virtual"}
{"actions":["action_name"],"subtype":"label"}

The subtype must be "real", "virtual", or "label". Omit subtype when none of
the three drift-adaptation actions is selected.

When no intervention should be active, return:

{"actions":["no_op"]}

Do not return reasoning, confidence, explanations, Markdown, or additional
keys."""


def _transient_api_error(exc) -> bool:
    status = getattr(exc, "status_code", None)
    return (
        status in {408, 409, 429} or
        isinstance(status, int) and status >= 500 or
        isinstance(exc, (TimeoutError, ConnectionError)) or
        type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError", "RateLimitError",
            "InternalServerError", "ConnectError", "ReadTimeout",
        }
    )


def make_openai_query_fn(model, base_url=None, api_key_env="OPENAI_API_KEY",
                         temperature=0.0, input_cost_per_million=0.0,
                         output_cost_per_million=0.0, system_prompt=SYSTEM_PROMPT,
                         timeout_seconds=120.0,
                         retry_delays_seconds=(2, 5, 10, 20, 40),
                         reasoning_effort=None):
    """OpenAI-compatible query function with token/cost metadata."""
    from openai import OpenAI
    retry_delays = tuple(float(value) for value in retry_delays_seconds)
    if any(value < 0 for value in retry_delays):
        raise ValueError("retry delays must be non-negative")
    client = OpenAI(api_key=os.environ.get(api_key_env) or "not-required",
                    base_url=base_url, timeout=float(timeout_seconds), max_retries=0)
    journal = {"path": None, "identity": {}, "records": {}}

    def configure_journal(path, identity):
        path = Path(path)
        identity = dict(identity)
        records = {}
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            lines = raw.splitlines()
            if raw and not raw.endswith("\n"):
                lines = lines[:-1]  # interrupted final append is safe to retry
            for line in lines:
                row = json.loads(line)
                if any(row.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"LLM journal identity mismatch: {path}")
                records[row["request_sha256"]] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        journal.update(path=path, identity=identity, records=records)

    def query_fn(text):
        query_fn.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        query_fn.last_cost_usd = 0.0
        query_fn.last_error = ""
        query_fn.last_attempts = 0
        query_fn.last_latency_ms = 0.0
        query_fn.last_replayed = False
        request_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        query_fn.last_request_sha256 = request_sha
        saved = journal["records"].get(request_sha)
        if saved is not None:
            query_fn.last_usage = saved["usage"]
            query_fn.last_cost_usd = float(saved["cost_usd"])
            query_fn.last_attempts = int(saved["attempts"])
            query_fn.last_latency_ms = float(saved["latency_ms"])
            query_fn.last_replayed = True
            return saved["raw_output"]

        started = time.perf_counter()
        for attempt in range(len(retry_delays) + 1):
            query_fn.last_attempts = attempt + 1
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": text}],
                    **({"extra_body": {"reasoning_effort": reasoning_effort}}
                       if reasoning_effort is not None else {}),
                )
                break
            except Exception as exc:
                if attempt == len(retry_delays) or not _transient_api_error(exc):
                    query_fn.last_error = type(exc).__name__
                    query_fn.last_latency_ms = (time.perf_counter() - started) * 1000
                    return '{"actions":["no_op"]}'
                time.sleep(retry_delays[attempt])

        usage = getattr(resp, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        query_fn.last_usage = {"prompt_tokens": prompt,
                               "completion_tokens": completion}
        if input_cost_per_million is not None and output_cost_per_million is not None:
            query_fn.last_cost_usd = (
                prompt * float(input_cost_per_million) +
                completion * float(output_cost_per_million)
            ) / 1_000_000
        query_fn.last_latency_ms = (time.perf_counter() - started) * 1000
        try:
            raw_output = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raw_output = None
        if journal["path"] is not None:
            try:
                round_idx = int(json.loads(text).get("round"))
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                round_idx = None
            record = {
                "schema_version": 1,
                **journal["identity"],
                "round": round_idx,
                "request_sha256": request_sha,
                "raw_output": raw_output,
                "usage": query_fn.last_usage,
                "cost_usd": query_fn.last_cost_usd,
                "latency_ms": query_fn.last_latency_ms,
                "attempts": query_fn.last_attempts,
            }
            with journal["path"].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            journal["records"][request_sha] = record
        return raw_output

    query_fn.model_name = model
    query_fn.configure_journal = configure_journal
    query_fn.backend_spec = {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "temperature": temperature,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "input_cost_per_million": input_cost_per_million,
        "output_cost_per_million": output_cost_per_million,
        "timeout_seconds": float(timeout_seconds),
        "retry_delays_seconds": list(retry_delays),
        "reasoning_effort": reasoning_effort,
    }
    query_fn.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    query_fn.last_cost_usd = 0.0
    query_fn.last_error = ""
    query_fn.last_attempts = 0
    query_fn.last_latency_ms = 0.0
    query_fn.last_replayed = False
    query_fn.last_request_sha256 = ""
    return query_fn


def load_model_specs(path, require_keys=True):
    """Load the shared formal-model roster without putting secrets in JSON."""
    with open(path, encoding="utf-8") as handle:
        models = json.load(handle).get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model config requires a non-empty models list")
    normalized, names = [], set()
    for row in models:
        if not isinstance(row, dict):
            raise ValueError("each model config entry must be an object")
        name, model = row.get("name"), row.get("model")
        if (not isinstance(name, str) or not name or
                any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for char in name)):
            raise ValueError("model name must be a filesystem-safe identifier")
        if name in names or not isinstance(model, str) or not model:
            raise ValueError("model names must be unique and model ids non-empty")
        names.add(name)
        api_key_env = row.get("api_key_env", "OPENAI_API_KEY")
        requires_key = bool(row.get("requires_api_key", True))
        if require_keys and requires_key and not os.environ.get(api_key_env):
            raise ValueError(f"missing API key environment variable: {api_key_env}")
        prices = [row.get("input_cost_per_million"),
                  row.get("output_cost_per_million")]
        if any(value is not None and (not isinstance(value, (int, float)) or value < 0)
               for value in prices):
            raise ValueError("model token prices must be null or non-negative")
        normalized.append({
            "name": name,
            "kind": row.get("kind", "model"),
            "model": model,
            "base_url": row.get("base_url"),
            "api_key_env": api_key_env,
            "requires_api_key": requires_key,
            "temperature": float(row.get("temperature", 0.0)),
            "input_cost_per_million": prices[0],
            "output_cost_per_million": prices[1],
            "reasoning_effort": row.get("reasoning_effort"),
        })
    return normalized


def make_qwen_query_fn(model="qwen3.8-max", temperature=0.0):
    """DashScope compatibility wrapper."""
    return make_openai_query_fn(model, BASE_URL, "DASHSCOPE_API_KEY", temperature)


if __name__ == "__main__":   # `python llm_backend.py` — confirm .env loaded a key
    k = os.environ.get("DASHSCOPE_API_KEY")
    print("DASHSCOPE_API_KEY loaded:", bool(k), f"(len={len(k)})" if k else "(not found)")
