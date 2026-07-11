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

    def __call__(self, url, json=None, headers=None, timeout=None):
        model = json["model"]
        self.calls.append((model, url))
        responses = self.script[model]
        return responses.pop(0) if len(responses) > 1 else responses[0]


def _make_fw(monkeypatch, models, post):
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://fw.example/v1")
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    monkeypatch.setenv("ALLOWED_MODELS", models)
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
