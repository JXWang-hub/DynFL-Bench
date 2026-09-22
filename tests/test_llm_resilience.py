"""Small runnable check for API retry, journal replay, and retryable checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import tempfile
from types import SimpleNamespace

import openai

import b3_composite as composite
from llm_backend import make_openai_query_fn


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.last_request = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.calls += 1
        self.last_request = request
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
        )


def make_query(client, retry_delays=(0, 0), reasoning_effort=None):
    original = openai.OpenAI
    openai.OpenAI = lambda **_: client
    try:
        return make_openai_query_fn(
            "fixture", timeout_seconds=1,
            retry_delays_seconds=retry_delays,
            reasoning_effort=reasoning_effort,
        )
    finally:
        openai.OpenAI = original


def history(error=""):
    return {
        "acc": [0.0],
        "hidden_event_log": [{}],
        "telemetry": [{
            "round": 0,
            "decision_due": True,
            "agent_interface_error": error,
            "llm_api_error": error,
            "llm_budget_error": "",
        }],
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        journal = root / "calls.jsonl"
        identity = {"run_fingerprint_sha256": "run", "agent_config_id": "agent"}
        client = FakeClient([TimeoutError(), TimeoutError(), '{"actions":["no_op"]}'])
        query = make_query(client)
        query.configure_journal(journal, identity)
        request = json.dumps({"round": 41})
        assert query(request) == '{"actions":["no_op"]}'
        assert client.calls == 3 and query.last_attempts == 3
        assert len(journal.read_text(encoding="utf-8").splitlines()) == 1

        direct_client = FakeClient(['{"actions":["no_op"]}'])
        direct = make_query(direct_client, reasoning_effort="none")
        assert direct(request) == '{"actions":["no_op"]}'
        assert direct_client.last_request["extra_body"] == {"reasoning_effort": "none"}
        assert direct.backend_spec["reasoning_effort"] == "none"

        replay_client = FakeClient([AssertionError("API must not be called")])
        replay = make_query(replay_client)
        replay.configure_journal(journal, identity)
        assert replay(request) == '{"actions":["no_op"]}'
        assert replay_client.calls == 0 and replay.last_replayed is True

        permanent_client = FakeClient([ValueError("bad model")])
        permanent = make_query(permanent_client)
        permanent.configure_journal(root / "permanent.jsonl", identity)
        assert permanent(request) == '{"actions":["no_op"]}'
        assert permanent_client.calls == 1 and permanent.last_error == "ValueError"

        checkpoint = root / "model.json"
        cfg = SimpleNamespace(seed=0, decide_every=1)
        data = {"b3_episode_plan": {"schema_version": 2}}
        spec = {"name": "model", "model": "fixture", "decision_cadence": 1}
        agent = SimpleNamespace(name="model")

        class FakeEngine:
            result = history("TimeoutError")

            def __init__(self, *_):
                self.current_round = 0

            def run(self, *_args, **_kwargs):
                return self.result

        original_engine = composite.Engine
        composite.Engine = FakeEngine
        try:
            first = composite._checkpoint_history(
                checkpoint, agent, cfg, data, "ep_fixture", "fingerprint",
                False, checkpoint_metadata=spec, audit_run_id="run_fixture",
            )
            assert first["telemetry"][0]["llm_api_error"] == "TimeoutError"
            assert not checkpoint.exists()
            assert len(list((root / "failed_attempts").glob("model_*.json"))) == 1

            FakeEngine.result = history()
            composite._checkpoint_history(
                checkpoint, agent, cfg, data, "ep_fixture", "fingerprint",
                True, checkpoint_metadata=spec, audit_run_id="run_fixture",
            )
            assert checkpoint.exists()
        finally:
            composite.Engine = original_engine

    print("LLM_RESILIENCE_OK")


if __name__ == "__main__":
    main()
