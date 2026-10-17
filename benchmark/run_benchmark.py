"""Compare deciders on the planted-flaw cases: verdict accuracy and steps to verdict."""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proofpath import (AdaptiveDecider, Claim, FixedOrderDecider,  # noqa: E402
                       ModelDecider, RandomDecider, run_agent)

CASES = ROOT / "data" / "cases"


def load_cases():
    return [json.loads(p.read_text()) for p in sorted(CASES.glob("*/meta.json"))]


def evaluate(make_decider, cases, runs=1):
    rows = []
    for case in cases:
        steps, correct = [], 0
        for r in range(runs):
            rep = run_agent(Claim.from_case(case, ROOT), make_decider(r))
            steps.append(len(rep.steps))
            correct += rep.verdict == case["expected"]
        rows.append((case["name"], correct / runs, statistics.mean(steps)))
    return rows


def main():
    cases = load_cases()
    if not cases:
        sys.exit("No cases found. Run: python benchmark/make_cases.py")
    strategies = {
        "random (avg of 30)": (lambda r: RandomDecider(seed=r), 30),
        "fixed checklist": (lambda r: FixedOrderDecider(), 1),
        "adaptive": (lambda r: AdaptiveDecider(), 1),
    }
    if os.environ.get("DECIDER_URL"):
        strategies["Decider 2B"] = (lambda r: ModelDecider(), 1)

    results = {k: evaluate(f, cases, runs) for k, (f, runs) in strategies.items()}
    names = [c["name"] for c in cases]
    print("\nSteps to verdict (lower is better)\n")
    print(f"{'case':<16}" + "".join(f"{k:>22}" for k in results))
    for i, n in enumerate(names):
        print(f"{n:<16}" + "".join(f"{results[k][i][2]:>22.1f}" for k in results))
    print(f"{'MEAN':<16}" + "".join(
        f"{statistics.mean(r[2] for r in results[k]):>22.2f}" for k in results))
    print(f"{'VERDICT ACC':<16}" + "".join(
        f"{statistics.mean(r[1] for r in results[k]):>22.0%}" for k in results))


if __name__ == "__main__":
    main()
