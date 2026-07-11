from verify import (check_summary, extract_code, generic_smoke_tests,
                    looks_confident, parse_summary_constraints, run_code_tests,
                    solve_assignment_puzzle, synthesize_tests)


def test_extract_code_fenced():
    text = "The bug is X.\n```python\ndef get_max(nums):\n    return max(nums)\n```\nDone."
    assert extract_code(text).startswith("def get_max")


def test_extract_code_bare_def():
    text = "def add(a, b):\n    return a + b"
    assert extract_code(text) == text


def test_run_code_tests_pass():
    code = "def second_largest(nums):\n    s = sorted(set(nums))\n    return s[-2]"
    tests = synthesize_tests("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.")
    assert tests
    report = run_code_tests(code, tests)
    assert report["ok"]
    assert all(r["passed"] for r in report["results"])


def test_run_code_tests_fail():
    code = "def second_largest(nums):\n    return sorted(nums)[-2]"  # duplicate bug
    tests = synthesize_tests("return the second-largest number in a list")
    report = run_code_tests(code, tests)
    assert not all(r.get("passed") for r in report["results"])


def test_run_code_tests_syntax_error():
    report = run_code_tests("def broken(:\n    pass", [])
    assert not report["ok"]


def test_run_code_tests_sandbox_timeout():
    report = run_code_tests("while True: pass", [], timeout=2)
    assert not report["ok"]


def test_generic_smoke_tests():
    assert generic_smoke_tests("def f(nums):\n    return nums")[0]["args"] == [[3, 1, 2]]
    assert generic_smoke_tests("def f(text):\n    return text")[0]["args"] == ["hello"]


def test_logic_solver_sample():
    prompt = ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
              "Sam does not own the bird. Jo owns the dog. Who owns the cat?")
    solved = solve_assignment_puzzle(prompt)
    assert solved is not None
    person, item, assign = solved
    assert person == "Sam" and item == "cat"
    assert assign == {"Sam": "cat", "Jo": "dog", "Lee": "bird"}


def test_logic_solver_rejects_unparseable():
    assert solve_assignment_puzzle("If all bloops are razzies, are all bloops lazzies?") is None


def test_summary_constraints():
    c = parse_summary_constraints("Summarize this in exactly one sentence: ...")
    assert c == {"sentences_exact": 1}
    ok, _ = check_summary("One sentence here.", c)
    assert ok
    ok, v = check_summary("Two sentences. Right here.", c)
    assert not ok and v

    c = parse_summary_constraints("Summarize in no more than 10 words.")
    assert c == {"words_max": 10}
    ok, _ = check_summary("Only five words in this.", c)
    assert ok
    ok, _ = check_summary(" ".join(["word"] * 11), c)
    assert not ok


def test_summary_abbreviations_not_sentence_breaks():
    c = {"sentences_exact": 1}
    ok, _ = check_summary("Revenue rose 3.5 percent, e.g. in Europe and Asia.", c)
    assert ok


def test_looks_confident():
    assert looks_confident("Canberra is the capital, near Lake Burley Griffin.")
    assert not looks_confident("I don't know the answer to that.")
    assert not looks_confident("")
