import copy

import pytest

import fireworks as fw_mod
from fireworks import Fireworks


class FakeResponse:
    def __init__(self, status, body_json=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._json = body_json
        self.text = text or (str(body_json) if body_json else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _ok_body(content="hi", tokens=42):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": tokens}}


class FakePost:
    """Scripted requests.post: maps model id -> list of responses (in order)."""
    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []  # (model, url)
        self.bodies = []
        self.timeouts = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        model = json["model"]
        self.calls.append((model, url))
        self.bodies.append(copy.deepcopy(json))
        self.timeouts.append(timeout)
        responses = self.script[model]
        return responses.pop(0) if len(responses) > 1 else responses[0]


def _make_fw(monkeypatch, models, post):
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://fw.example/v1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    monkeypatch.setenv("ALLOWED_MODELS", models)
    monkeypatch.delenv("FIREWORKS_EXTRA_BODY", raising=False)
    monkeypatch.delenv("PREFERRED_MODEL_HINTS", raising=False)
    monkeypatch.setattr(fw_mod.requests, "post", post)
    monkeypatch.setattr(fw_mod.time, "sleep", lambda s: None)
    return Fireworks()


def test_picks_smallest_model(monkeypatch):
    fw = _make_fw(monkeypatch, "acct/llama-70b-instruct,acct/llama-1b-instruct",
                  FakePost({}))
    assert fw.model == "acct/llama-1b-instruct"


def test_404_falls_back_and_caches_bad_model(monkeypatch):
    post = FakePost({
        "acct/llama-1b-instruct": [FakeResponse(404, text='{"error":"Model not found"}')],
        "acct/llama-70b-instruct": [FakeResponse(200, _ok_body("answer", 50))],
    })
    fw = _make_fw(monkeypatch, "acct/llama-70b-instruct,acct/llama-1b-instruct", post)
    assert fw.complete("q", 100) == "answer"
    # 404 -> non-retryable: exactly one call to the bad model, then the fallback.
    assert [m for m, _ in post.calls] == ["acct/llama-1b-instruct", "acct/llama-70b-instruct"]
    assert "acct/llama-1b-instruct" in fw.bad_models
    assert fw.model == "acct/llama-70b-instruct"
    # Known-bad is cached: the next call never touches the 404 model again.
    post.calls.clear()
    assert fw.complete("q2", 100) == "answer"
    assert [m for m, _ in post.calls] == ["acct/llama-70b-instruct"]


def test_all_models_404_disables_client(monkeypatch):
    post = FakePost({"acct/a-1b": [FakeResponse(404)], "acct/b-8b": [FakeResponse(404)]})
    fw = _make_fw(monkeypatch, "acct/a-1b,acct/b-8b", post)
    assert fw.complete("q", 100) == ""
    assert not fw.available
    # Dead client short-circuits without HTTP calls.
    post.calls.clear()
    assert fw.complete("q", 100) == ""
    assert post.calls == []


def test_5xx_retried_once_then_gives_up(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(500, text="oops")]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100) == ""
    assert len(post.calls) == 2  # original + single retry


def test_400_not_retried(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(400, text="bad request")]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100) == ""
    assert len(post.calls) == 1


def test_counters_attempted_vs_succeeded_and_tokens(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(500),
                                   FakeResponse(200, _ok_body("a", 30)),
                                   FakeResponse(200, _ok_body("b", 70))]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q1", 100) == "a"   # 500 then 200 on retry
    assert fw.complete("q2", 100) == "b"
    assert fw.calls_attempted == 3
    assert fw.calls_succeeded == 2
    assert fw.total_tokens == 100


def test_usage_total_is_derived_or_marked_unknown_without_losing_answer(
        monkeypatch):
    post = FakePost({"acct/m-1b": [
        FakeResponse(200, {
            "choices": [{"message": {"content": "derived"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }),
        FakeResponse(200, {
            "choices": [{"message": {"content": "unknown"}}],
            "usage": {"total_tokens": "not-a-number"},
        }),
    ]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)

    assert fw.complete("q1", 100) == "derived"
    assert fw.complete("q2", 100) == "unknown"
    assert fw.total_tokens == 18
    assert fw.derived_usage_calls == 1
    assert fw.unknown_usage_calls == 1


def test_requests_uses_bounded_connect_and_read_timeouts(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(200, _ok_body())]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100, timeout=9.0) == "hi"
    connect, read = post.timeouts[0]
    assert 0 < connect <= 3.0
    assert 0 < read < 9.0
    assert connect + read < 9.0


def test_empty_2xx_content_is_retried(monkeypatch):
    post = FakePost({"acct/m-1b": [
        FakeResponse(200, _ok_body("", 10)),
        FakeResponse(200, _ok_body("answer", 20)),
    ]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100) == "answer"
    assert fw.calls_attempted == 2
    assert fw.calls_succeeded == 1
    assert fw.total_tokens == 30


def test_response_normalizes_unicode_spacing(monkeypatch):
    post = FakePost({"acct/m-1b": [
        FakeResponse(200, _ok_body("Andy\u202fJassy\u2003-\u202fPerson", 20)),
    ]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100) == "Andy Jassy - Person"


def test_response_preserves_python_indentation(monkeypatch):
    code = "```python\ndef f():\n    return 1\n```"
    post = FakePost({"acct/m-1b": [FakeResponse(200, _ok_body(code, 20))]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert "\n    return 1" in fw.complete("q", 100)


def test_direct_answer_model_preferred_over_flash(monkeypatch):
    fw = _make_fw(monkeypatch,
                  "acct/gemma-4-31b-it,acct/gpt-oss-20b,acct/deepseek-v4-flash",
                  FakePost({}))
    assert fw.model == "acct/gpt-oss-20b"


def test_gpt_oss_preferred_for_separate_final_content(monkeypatch):
    fw = _make_fw(monkeypatch,
                  "acct/deepseek-v4-flash,acct/gpt-oss-20b,acct/deepseek-v4-pro",
                  FakePost({}))
    assert fw.model == "acct/gpt-oss-20b"


def test_ordered_empirical_hints_put_pro_before_generic_fallback(monkeypatch):
    monkeypatch.setenv(
        "PREFERRED_MODEL_HINTS", "deepseek-v4-flash,deepseek-v4-pro")
    fw = _make_fw(
        monkeypatch,
        "acct/gpt-oss-20b,acct/deepseek-v4-pro,acct/deepseek-v4-flash",
        FakePost({}),
    )
    # _make_fw intentionally clears inherited preferences; set them after and
    # repick to exercise the image's ordered build-time setting.
    monkeypatch.setenv(
        "PREFERRED_MODEL_HINTS", "deepseek-v4-flash,deepseek-v4-pro")
    fw.model = fw._pick_model()
    assert fw.model == "acct/deepseek-v4-flash"
    fw.bad_models.add("acct/deepseek-v4-flash")
    assert fw._pick_model() == "acct/deepseek-v4-pro"


@pytest.mark.parametrize("raw", ["{broken", "[]", '"text"', "null"])
def test_extra_body_rejects_malformed_or_non_object_values(monkeypatch, raw):
    monkeypatch.setenv("FIREWORKS_EXTRA_BODY", raw)
    assert fw_mod._extra_body() == {}


def test_extra_body_whitelist_cannot_override_protected_request_fields(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(200, _ok_body("answer"))]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    monkeypatch.setenv("FIREWORKS_EXTRA_BODY", '''{
        "reasoning_effort": "none",
        "model": "acct/not-allowed",
        "messages": [{"role": "user", "content": "replaced"}],
        "max_tokens": 9999,
        "temperature": 2,
        "api_key": "must-not-leak",
        "Authorization": "must-not-leak"
    }''')

    assert fw.complete("real prompt", 123, system="real system") == "answer"
    body = post.bodies[0]
    assert body == {
        "reasoning_effort": "none",
        "model": "acct/m-1b",
        "messages": [
            {"role": "system", "content": "real system"},
            {"role": "user", "content": "real prompt"},
        ],
        "temperature": 0,
        "max_tokens": 123,
    }


def test_unsupported_extra_param_retry_drops_only_safe_extras(monkeypatch):
    post = FakePost({"acct/m-1b": [
        FakeResponse(400, text="unsupported reasoning option"),
        FakeResponse(200, _ok_body("answer")),
    ]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    monkeypatch.setenv(
        "FIREWORKS_EXTRA_BODY",
        '{"reasoning_effort":"none","model":"acct/not-allowed"}',
    )

    assert fw.complete("q", 77) == "answer"
    assert len(post.bodies) == 2
    assert post.bodies[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in post.bodies[1]
    for body in post.bodies:
        assert body["model"] == "acct/m-1b"
        assert body["messages"] == [{"role": "user", "content": "q"}]
        assert body["max_tokens"] == 77
        assert body["temperature"] == 0

    # The rejection is cached per model; later calls start without the known-
    # unsupported option instead of paying for another HTTP 400.
    assert fw.complete("q2", 77) == "answer"
    assert "reasoning_effort" not in post.bodies[2]


def test_auth_failure_marks_provider_definitively_unavailable(monkeypatch):
    post = FakePost({"acct/m-1b": [FakeResponse(401, text="bad key")]})
    fw = _make_fw(monkeypatch, "acct/m-1b", post)
    assert fw.complete("q", 100) == ""
    assert fw.definitive_unavailable is True
    assert fw.available is False
