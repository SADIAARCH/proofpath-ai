from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Claim
from .decider import Decider, State
from .tools import TOOLS, Context, Finding, FAIL, OK, WARN

CORE = ["recompute_metric", "class_balance", "sample_size", "per_class"]


@dataclass
class Step:
    index: int
    action: str
    rationale: str
    finding: Finding


@dataclass
class Report:
    verdict: str                     # supported | contradicted | insufficient_evidence
    headline: str
    steps: list = field(default_factory=list)
    unverified: list = field(default_factory=list)   # "check (needs artifact)"
    coverage: float = 0.0            # share of all tools actually run


def _verdict(steps, unverified):
    done = {s.action: s.finding for s in steps}
    fails = [s for s in steps if s.finding.status == FAIL]
    if fails:
        return "contradicted", fails[0].finding.summary
    warns = [s for s in steps if s.finding.status == WARN]
    if warns:
        return "insufficient_evidence", warns[0].finding.summary
    if all(c in done and done[c].status == OK for c in CORE):
        caveat = f" Not verified: {', '.join(u.split(' (')[0] for u in unverified)}." if unverified else ""
        return "supported", "Every applicable check passed." + caveat
    return "insufficient_evidence", "Not enough artifacts to check the core questions."


def run_agent(claim: Claim, decider: Decider, max_steps: int = 10, stop_on_fail: bool = True) -> Report:
    ctx = Context(claim)
    remaining = [n for n, t in TOOLS.items() if t.available(ctx)]
    unverified = [f"{t.label} (needs {t.needs})" for n, t in TOOLS.items() if n not in remaining]
    steps: list[Step] = []
    findings: list[Finding] = []
    while remaining and len(steps) < max_steps:
        action, why = decider.choose(State(ctx, findings, list(remaining)))
        if action not in remaining:
            action, why = remaining[0], "invalid choice, defaulted to first remaining"
        finding = TOOLS[action].func(ctx)
        remaining.remove(action)
        findings.append(finding)
        steps.append(Step(len(steps) + 1, action, why, finding))
        if stop_on_fail and finding.status == FAIL:
            break
    verdict, headline = _verdict(steps, unverified)
    return Report(verdict, headline, steps, unverified, len(steps) / len(TOOLS))
