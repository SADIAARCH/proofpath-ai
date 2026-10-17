import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proofpath import (AdaptiveDecider, BedrockDecider, Claim, FixedOrderDecider,
                       ModelDecider, run_agent)  # noqa: E402
from proofpath.tools import wilson_interval  # noqa: E402


@pytest.fixture(scope="session")
def cases():
    subprocess.run([sys.executable, str(ROOT / "benchmark" / "make_cases.py")], check=True)
    return [json.loads(p.read_text()) for p in sorted((ROOT / "data" / "cases").glob("*/meta.json"))]


def test_adaptive_gets_every_verdict_right(cases):
    for case in cases:
        rep = run_agent(Claim.from_case(case, ROOT), AdaptiveDecider())
        assert rep.verdict == case["expected"], case["name"]


def test_fixed_gets_every_verdict_right(cases):
    for case in cases:
        rep = run_agent(Claim.from_case(case, ROOT), FixedOrderDecider())
        assert rep.verdict == case["expected"], case["name"]


def test_adaptive_needs_fewer_steps_than_fixed(cases):
    a = sum(len(run_agent(Claim.from_case(c, ROOT), AdaptiveDecider()).steps) for c in cases)
    f = sum(len(run_agent(Claim.from_case(c, ROOT), FixedOrderDecider()).steps) for c in cases)
    assert a < f


def test_model_decider_falls_back_on_bad_answer(cases):
    case = next(c for c in cases if c["name"] == "imbalance")
    rep = run_agent(Claim.from_case(case, ROOT), ModelDecider(client=lambda p, c: "nonsense"))
    assert rep.verdict == "contradicted"


def test_model_decider_follows_valid_answer(cases):
    case = next(c for c in cases if c["name"] == "clean")
    rep = run_agent(Claim.from_case(case, ROOT), ModelDecider(client=lambda p, c: "leakage"))
    assert rep.steps[0].action == "leakage"


def test_bedrock_decider_falls_back_without_credentials(cases):
    case = next(c for c in cases if c["name"] == "imbalance")
    rep = run_agent(
        Claim.from_case(case, ROOT),
        BedrockDecider(client=lambda p, c: (_ for _ in ()).throw(RuntimeError("offline"))),
    )
    assert rep.verdict == "contradicted"


def test_bedrock_decider_accepts_a_tool_choice(cases):
    case = next(c for c in cases if c["name"] == "clean")
    rep = run_agent(
        Claim.from_case(case, ROOT),
        BedrockDecider(client=lambda p, c: "leakage"),
    )
    assert rep.steps[0].action == "leakage"


def test_wilson_interval_shrinks_with_n():
    lo1, hi1 = wilson_interval(0.9, 40)
    lo2, hi2 = wilson_interval(0.9, 4000)
    assert (hi1 - lo1) > (hi2 - lo2)
