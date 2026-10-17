"""Deciders choose the NEXT evidence action. Swap in Decider 2B via ModelDecider."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import urllib.request
from dataclasses import dataclass

from botocore.exceptions import BotoCoreError, ClientError

from .tools import TOOLS, Context, Finding, OK


@dataclass
class State:
    ctx: Context
    findings: list          # list[Finding] gathered so far
    remaining: list         # list[str] action names still available


class Decider:
    name = "base"

    def choose(self, state: State) -> tuple[str, str]:
        raise NotImplementedError


class FixedOrderDecider(Decider):
    """Baseline: a static checklist, always the same order."""
    name = "fixed checklist"
    ORDER = ["paper_check", "eval_method", "sample_size", "class_balance",
             "recompute_metric", "per_class", "leakage"]

    def choose(self, state):
        for a in self.ORDER:
            if a in state.remaining:
                return a, "next item on the fixed checklist"
        return state.remaining[0], "fallback"


class RandomDecider(Decider):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state):
        return self.rng.choice(state.remaining), "random pick"


_TUNING_HINT = re.compile(r"(tun|select|chos|optimi|pick)\w*\W+(\w+\W+){0,8}test|test\W+(\w+\W+){0,4}(tun|select)", re.I)

_BASE = {"class_balance": 3, "recompute_metric": 2.5, "sample_size": 1.5,
         "paper_check": 1.5, "eval_method": 1.0, "per_class": 1.0, "leakage": 1.0}
_DEFAULT_WHY = {
    "class_balance": "cheap check that rules out a trivial majority-class baseline",
    "recompute_metric": "verify the number against the released predictions",
    "sample_size": "check whether the test set is big enough to support the number",
    "paper_check": "cross-check the claim against what the paper reports",
    "eval_method": "read how the evaluation was run",
    "per_class": "look for classes the model gets wrong",
    "leakage": "check whether test rows leaked into training",
}


class AdaptiveDecider(Decider):
    """Value-of-information heuristic: score each remaining action using the claim,
    cheap metadata, and what earlier findings revealed. This is the stand-in policy
    that Decider 2B replaces (see ModelDecider)."""
    name = "adaptive (heuristic)"

    def choose(self, state):
        done = {f.tool: f for f in state.findings}
        best = None
        for idx, name in enumerate(state.remaining):
            pts, why = self._score(name, state.ctx, done)
            key = (pts, -idx)
            if best is None or key > best[0]:
                best = (key, name, why)
        return best[1], best[2]

    @staticmethod
    def _score(name, ctx, done):
        pts, why = _BASE[name], _DEFAULT_WHY[name]
        c = ctx.claim
        high = c.value >= 0.95
        if name == "class_balance" and high:
            pts += 2; why = "very high claimed accuracy: first rule out a majority-class baseline"
        if name == "sample_size" and ctx.n_rows is not None and ctx.n_rows < 200:
            pts += 4.5; why = f"only {ctx.n_rows} test rows, the number may be noise"
        if name == "eval_method" and _TUNING_HINT.search(c.eval_notes):
            pts += 4.5; why = "evaluation notes hint that the test set influenced model choices"
        if name == "leakage":
            if high:
                pts += 2; why = "high accuracy with training data available: check train/test overlap"
            if c.value >= 0.99:
                pts += 1
        if name == "per_class" and "class_balance" in done and done["class_balance"].status != OK:
            pts += 4; why = "class balance looked off, so check per-class recall"
        if name == "paper_check":
            if "recompute_metric" in done and done["recompute_metric"].status != OK:
                pts += 4; why = "recomputed metric disagrees, see what the paper reports"
            if ctx.preds is None:
                pts += 3; why = "no predictions available, the paper is the only evidence"
        return pts, why


class ModelDecider(Decider):
    """Calls Decider 2B (or any chooser) with a compact state summary.

    Set DECIDER_URL to an HTTP endpoint that accepts
        {"prompt": str, "choices": [str, ...]}
    and returns {"choice": str}. If your Decider 2B deployment has a different
    interface, edit `_default_client` only. Invalid or failed answers fall back
    to the heuristic so the demo never stalls.
    """
    name = "Decider 2B"

    def __init__(self, client=None, fallback: Decider | None = None):
        self.client = client or self._default_client
        self.fallback = fallback or AdaptiveDecider()

    @staticmethod
    def _default_client(prompt: str, choices: list[str]) -> str:
        url = os.environ["DECIDER_URL"]
        req = urllib.request.Request(
            url, data=json.dumps({"prompt": prompt, "choices": choices}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read())
        return str(body.get("choice") or body.get("text", "")).strip()

    @staticmethod
    def build_prompt(state: State) -> str:
        c = state.ctx.claim
        lines = [f"Claim: {c.text or c.metric} = {c.value:.1%}.",
                 f"Test rows: {state.ctx.n_rows}. Training data: {'yes' if state.ctx.train is not None else 'no'}.",
                 "Evidence so far:"]
        lines += [f"- {f.tool}: {f.status}. {f.summary}" for f in state.findings] or ["- none"]
        lines.append("Pick the single most informative next check from: " + ", ".join(state.remaining))
        return "\n".join(lines)

    def choose(self, state):
        try:
            answer = self.client(self.build_prompt(state), list(state.remaining)).strip().lower()
            for name in state.remaining:
                if name == answer or name in answer:
                    return name, "chosen by Decider 2B"
        except Exception as exc:  # network error, bad JSON, missing env var
            action, _ = self.fallback.choose(state)
            return action, f"Decider unavailable ({type(exc).__name__}), heuristic used"
        action, why = self.fallback.choose(state)
        return action, "Decider answer invalid, heuristic used: " + why


class StrandsDecider(Decider):
    """Use the official local ``strands-decider`` CLI for tool selection.

    This is the reusable hackathon decision pattern: a typed choice question
    routes the next evidence tool, while deterministic tools own all findings.
    """
    name = "Strands Decider 2B"

    def __init__(self, client=None, fallback: Decider | None = None,
                 model="StrandsAgents/strands-decider-2B-hobson-v19"):
        self.client = client or self._default_client
        self.fallback = fallback or AdaptiveDecider()
        self.model = model

    def _default_client(self, state_text: str, choices: list[str]) -> str:
        question = "Which evidence check should run next?"
        choice = f"{question}=" + ",".join(choices)
        result = subprocess.run(
            ["strands-decider", "ask", self.model, "--state", state_text,
             "--choice", choice],
            capture_output=True, text=True, timeout=30, check=True,
        )
        match = re.search(r"->\s*([A-Za-z0-9_]+)", result.stdout)
        if not match:
            raise ValueError("strands-decider returned no choice")
        return match.group(1)

    def choose(self, state):
        try:
            answer = self.client(
                ModelDecider.build_prompt(state), list(state.remaining)
            ).strip().lower()
            for name in state.remaining:
                if answer == name or name in answer:
                    return name, "chosen by Strands Decider 2B"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            action, _ = self.fallback.choose(state)
            return action, f"Strands Decider unavailable ({type(exc).__name__}), heuristic used"
        action, why = self.fallback.choose(state)
        return action, "Strands Decider answer invalid, heuristic used: " + why


class StrandsBedrockDecider(Decider):
    """Pair Decider 2B tool routing with a Bedrock explanation call."""
    name = "Strands Decider 2B + Bedrock"

    def __init__(self, strands=None, bedrock=None):
        self.strands = strands or StrandsDecider()
        self.bedrock = bedrock or BedrockDecider()

    def choose(self, state):
        action, rationale = self.strands.choose(state)
        try:
            explanation = self.bedrock.explain(
                f"Explain in one short sentence why '{action}' is the next "
                f"evidence check for this state:\n{ModelDecider.build_prompt(state)}"
            )
            return action, f"{rationale}. Bedrock: {explanation}"
        except (OSError, ValueError, KeyError, TypeError, BotoCoreError, ClientError) as exc:
            return action, f"{rationale}. Bedrock explanation unavailable ({type(exc).__name__})"


class BedrockDecider(Decider):
    """Use an Amazon Bedrock foundation model to choose the next check.

    The model only selects among deterministic evidence tools. It never
    produces the verdict, and any AWS/configuration failure falls back to the
    local adaptive policy.

    Configure with ``AWS_REGION`` and optionally ``BEDROCK_MODEL_ID``. The
    default model is Amazon Nova Lite, which is inexpensive for a live demo.
    """
    name = "Amazon Bedrock"

    def __init__(self, client=None, fallback: Decider | None = None,
                 model_id: str | None = None):
        self.client = client or self._default_client
        self.fallback = fallback or AdaptiveDecider()
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")

    def _default_client(self, prompt: str, choices: list[str]) -> str:
        import boto3

        runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        response = runtime.converse(
            modelId=self.model_id,
            system=[{"text": (
                "You are the routing brain for ProofPath, an evidence "
                "verification agent. Choose exactly one available tool name. "
                "Do not judge the claim and do not explain your answer."
            )}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 30},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return " ".join(str(part.get("text", "")) for part in content).strip()

    def explain(self, prompt: str) -> str:
        runtime = __import__("boto3").client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        response = runtime.converse(
            modelId=self.model_id,
            system=[{"text": "Answer concisely and do not invent evidence."}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 60},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = " ".join(str(part.get("text", "")) for part in content).strip()
        if not text:
            raise ValueError("Bedrock returned an empty explanation")
        return text

    def choose(self, state):
        try:
            answer = self.client(
                ModelDecider.build_prompt(state), list(state.remaining)
            ).strip().lower()
            for name in state.remaining:
                if answer == name or name in answer:
                    return name, f"chosen by Amazon Bedrock ({self.model_id})"
        except Exception as exc:  # SDK, credentials, network, or response errors
            action, _ = self.fallback.choose(state)
            return action, f"Bedrock unavailable ({type(exc).__name__}), heuristic used"
        action, why = self.fallback.choose(state)
        return action, "Bedrock answer invalid, heuristic used: " + why
