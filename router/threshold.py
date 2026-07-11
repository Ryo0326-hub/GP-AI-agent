"""Threshold tuning + routing-policy derivation. Pure stdlib on purpose:
unit-testable without torch, and reused by the projection tooling.

Cost model (asymmetric, from the leaderboard math):
  - a false "local_ok" that survives verification burns one of only ~2 spare
    misses against the 16/19 accuracy gate;
  - a false "escalate" merely costs ~100-300 Fireworks tokens.
So the threshold is chosen to minimize expected tokens SUBJECT TO a hard cap
on expected misses per 19 tasks, not to maximize plain accuracy.
"""

# Runtime acceptance of a local answer: verification passed, or the category
# is trusted enough to skip the soft gate.  Trust uses a confidence bound, not
# a point estimate: ten lucky examples are not evidence of 90% reliability.
TRUST_ACC = 0.90
ALWAYS_ESCALATE_ACC = 0.70
# Category-wide trust bypasses verification, so it needs direct evidence about
# unverified answers themselves. Overall accuracy is not a valid substitute:
# a category can look excellent because its verified answers are easy while
# its small unverified tail is consistently wrong.
MIN_UNVERIFIED_SAMPLES = 10
# Two-sided 95% Wilson interval.  Requiring its lower bound to clear 90%
# deliberately makes trust_local rare (40/40 clears; 10/10 does not).
WILSON_Z = 1.96

# Reactive escalation shares the 25 second end-to-end task allowance with the
# local attempt.  A remote call with less than eight seconds left is treated as
# unsafe during calibration; measured labels already contain local latency.
REACTIVE_DEADLINE_S = 25.0
MIN_REMOTE_WINDOW_S = 8.0
MIN_SLOW_FAILURE_RECALL = 0.90


def wilson_lower_bound(successes, total, z=WILSON_Z):
    """Wilson score lower bound for a Bernoulli success probability."""
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    center = p + z2 / (2.0 * total)
    margin = z * ((p * (1.0 - p) / total
                   + z2 / (4.0 * total * total)) ** 0.5)
    return (center - margin) / (1.0 + z2 / total)


def derive_policy(records):
    """Per-category routing policy from labeling records.

    always_escalate: local accuracy < 70% -> never trust the router to keep
      the category local.
    trust_local: at least MIN_UNVERIFIED_SAMPLES unverified answers were
      observed and those answers are >= 90% correct -> accept unverified local
      answers instead of escalating them. Sparse evidence never enables this
      accuracy-risking shortcut.
    """
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    policy = {"always_escalate": [], "trust_local": []}
    for cat, rs in sorted(by_cat.items()):
        acc = sum(r["correct"] for r in rs) / len(rs)
        if acc < ALWAYS_ESCALATE_ACC:
            policy["always_escalate"].append(cat)
            continue
        unver = [r for r in rs if not r["verified"]]
        if len(unver) >= MIN_UNVERIFIED_SAMPLES:
            correct = sum(bool(r["correct"]) for r in unver)
            if wilson_lower_bound(correct, len(unver)) >= TRUST_ACC:
                policy["trust_local"].append(cat)
    return policy


def is_slow_failure(record, reactive_deadline_s=REACTIVE_DEADLINE_S,
                    min_remote_window_s=MIN_REMOTE_WINDOW_S):
    """Whether a failed, verifier-bounced local attempt leaves too little time.

    Such examples must be routed up front.  Otherwise the runtime reaches the
    same Fireworks call only after consuming most of its end-to-end allowance.
    """
    latency = float(record.get("latency_s") or 0.0)
    return (not bool(record["correct"])
            and not bool(record["verified"])
            and latency > reactive_deadline_s - min_remote_window_s)


def policy_safety_metrics(records, threshold, policy,
                          reactive_deadline_s=REACTIVE_DEADLINE_S,
                          min_remote_window_s=MIN_REMOTE_WINDOW_S):
    """Observed routing-safety metrics for already-scored records."""
    failures = 0
    planned_failures = 0
    final_escalated_failures = 0
    slow_failures = 0
    slow_planned = 0
    unsafe_local = 0
    always = set(policy.get("always_escalate") or ())
    trust = set(policy.get("trust_local") or ())
    for record in records:
        planned = (record["category"] in always
                   or record["score"] >= threshold)
        reactive = (not planned and not record["verified"]
                    and record["category"] not in trust)
        if not planned and not reactive and not record["correct"]:
            unsafe_local += 1
        if not record["correct"]:
            failures += 1
            planned_failures += int(planned)
            final_escalated_failures += int(planned or reactive)
            slow = is_slow_failure(
                record, reactive_deadline_s, min_remote_window_s)
            slow_failures += int(slow)
            slow_planned += int(slow and planned)

    def ratio(numerator, denominator, empty=1.0):
        return numerator / denominator if denominator else empty

    count = len(records)
    return {
        "count": count,
        "failureCount": failures,
        "failurePlannedRecall": round(ratio(
            planned_failures, failures), 4),
        "failureFinalEscalationRecall": round(ratio(
            final_escalated_failures, failures), 4),
        "slowFailureCount": slow_failures,
        "slowFailurePlannedRecall": round(ratio(
            slow_planned, slow_failures), 4),
        "unsafeLocalCount": unsafe_local,
        "unsafeLocalRate": round(unsafe_local / count, 4) if count else 0.0,
    }


def simulate(records, threshold, policy, eval_size=19,
             esc_tokens=180, esc_accuracy=0.95,
             reactive_deadline_s=REACTIVE_DEADLINE_S,
             min_remote_window_s=MIN_REMOTE_WINDOW_S):
    """Project eval-time outcomes for scored holdout records at a threshold.

    Each record needs: score (P(escalate)), correct, verified, category.
    Returns expected misses and Fireworks tokens scaled to `eval_size` tasks.
    """
    n = len(records)
    if n == 0:
        raise ValueError("no records to simulate")
    misses = 0.0
    escalations = 0.0
    local_kept = 0
    planned_escalations = 0
    reactive_escalations = 0
    reactive_at_risk = 0
    fallback_misses = 0.0
    slow_failures = 0
    slow_failures_planned = 0
    for r in records:
        cat = r["category"]
        slow_failure = is_slow_failure(
            r, reactive_deadline_s, min_remote_window_s)
        if slow_failure:
            slow_failures += 1
        routed_escalate = (cat in policy["always_escalate"]
                           or r["score"] >= threshold)
        if routed_escalate:
            planned_escalations += 1
            if slow_failure:
                slow_failures_planned += 1
            escalations += 1
            misses += 1.0 - esc_accuracy
            continue
        if not routed_escalate:
            accepted = r["verified"] or cat in policy["trust_local"]
            if accepted:
                local_kept += 1
                if not r["correct"]:
                    misses += 1.0
                continue
            # Verification bounced it. Count the paid call either way, but do
            # not pretend a call with almost no remaining wall time has normal
            # remote-model accuracy.
            escalations += 1
            reactive_escalations += 1
            latency = float(r.get("latency_s") or 0.0)
            if latency > reactive_deadline_s - min_remote_window_s:
                reactive_at_risk += 1
                miss = 0.0 if r["correct"] else 1.0
                fallback_misses += miss
                misses += miss
            else:
                misses += 1.0 - esc_accuracy
    scale = eval_size / n
    slow_recall = (slow_failures_planned / slow_failures
                   if slow_failures else 1.0)
    return {
        "threshold": threshold,
        "expected_misses": round(misses * scale, 3),
        "expected_escalations": round(escalations * scale, 2),
        "expected_tokens": round(escalations * scale * esc_tokens),
        "local_share": round(local_kept / n, 3),
        "projected_accuracy": round(1.0 - (misses / n), 4),
        "planned_escalations": round(planned_escalations * scale, 2),
        "reactive_escalations": round(reactive_escalations * scale, 2),
        "reactive_at_risk": round(reactive_at_risk * scale, 2),
        "expected_fallback_misses": round(fallback_misses * scale, 3),
        "slow_failures": slow_failures,
        "slow_failures_planned": slow_failures_planned,
        "slow_failure_recall": round(slow_recall, 4),
        "policy_adjusted": True,
    }


def pick_threshold(records, policy, max_expected_misses=1.0, eval_size=19,
                   esc_tokens=180, esc_accuracy=0.95,
                   min_slow_failure_recall=MIN_SLOW_FAILURE_RECALL,
                   reactive_deadline_s=REACTIVE_DEADLINE_S,
                   min_remote_window_s=MIN_REMOTE_WINDOW_S):
    """Sweep thresholds; return (threshold, projection).

    Minimizes expected tokens among thresholds whose projected misses per
    `eval_size` tasks stay under the cap; if none qualify, minimizes misses.
    """
    candidates = sorted({r["score"] for r in records} | {0.0, 0.5, 1.01})
    best_feasible, best_any = None, None
    for t in candidates:
        proj = simulate(
            records, t, policy, eval_size, esc_tokens, esc_accuracy,
            reactive_deadline_s, min_remote_window_s)
        if best_any is None or \
                (-proj["slow_failure_recall"], proj["expected_misses"],
                 proj["expected_tokens"]) < \
                (-best_any["slow_failure_recall"], best_any["expected_misses"],
                 best_any["expected_tokens"]):
            best_any = proj
        if (proj["expected_misses"] <= max_expected_misses
                and proj["slow_failure_recall"] >= min_slow_failure_recall):
            if best_feasible is None or \
                    (proj["expected_tokens"], proj["expected_misses"]) < \
                    (best_feasible["expected_tokens"], best_feasible["expected_misses"]):
                best_feasible = proj
    chosen = best_feasible or best_any
    return chosen["threshold"], chosen
