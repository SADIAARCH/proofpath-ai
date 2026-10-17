"""Deterministic evidence tools. Each one computes something from real artifacts;
none of them asks a language model whether the claim is true."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .claims import Claim

OK, WARN, FAIL, NA = "ok", "warn", "fail", "unavailable"


@dataclass
class Finding:
    tool: str
    status: str
    summary: str
    data: dict = field(default_factory=dict)


class Context:
    """Lazily loads artifacts once and shares them across tools."""

    def __init__(self, claim: Claim):
        self.claim = claim
        self._preds: Optional[pd.DataFrame] = None
        self._train: Optional[pd.DataFrame] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        c = self.claim
        if c.predictions_path:
            df = pd.read_csv(c.predictions_path)
            if {"y_true", "y_pred"} <= set(df.columns):
                self._preds = df
        if c.train_path:
            df = pd.read_csv(c.train_path)
            if "y_true" in df.columns:
                self._train = df

    @property
    def preds(self) -> Optional[pd.DataFrame]:
        self._load()
        return self._preds

    @property
    def train(self) -> Optional[pd.DataFrame]:
        self._load()
        return self._train

    @property
    def n_rows(self) -> Optional[int]:
        return None if self.preds is None else len(self.preds)


def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ----------------------------------------------------------------- tools
def class_balance(ctx: Context) -> Finding:
    y = ctx.preds["y_true"]
    shares = y.value_counts(normalize=True)
    baseline = float(shares.max())
    gap = ctx.claim.value - baseline
    data = {"majority_share": round(baseline, 4), "gap": round(gap, 4),
            "classes": int(len(shares))}
    msg = (f"Always predicting the majority class scores {baseline:.1%}; "
           f"the claim beats it by {gap:+.1%}.")
    if gap < 0.02:
        return Finding("class_balance", FAIL, msg + " The claim shows no skill.", data)
    if gap < 0.05:
        return Finding("class_balance", WARN, msg + " Margin is thin.", data)
    return Finding("class_balance", OK, msg, data)


def recompute_metric(ctx: Context) -> Finding:
    p = ctx.preds
    acc = float((p["y_true"] == p["y_pred"]).mean())
    diff = abs(acc - ctx.claim.value)
    data = {"recomputed": round(acc, 4), "claimed": ctx.claim.value, "diff": round(diff, 4)}
    msg = f"Recomputed accuracy is {acc:.1%} versus the claimed {ctx.claim.value:.1%}."
    if diff > 0.01:
        return Finding("recompute_metric", FAIL, msg, data)
    if diff > 0.005:
        return Finding("recompute_metric", WARN, msg, data)
    return Finding("recompute_metric", OK, msg + " Matches.", data)


def sample_size(ctx: Context) -> Finding:
    n = len(ctx.preds)
    lo, hi = wilson_interval(ctx.claim.value, n)
    width = hi - lo
    data = {"n": n, "ci_low": round(lo, 4), "ci_high": round(hi, 4), "width": round(width, 4)}
    msg = f"{n} test rows give a 95% interval of {lo:.1%} to {hi:.1%}."
    if width > 0.10 or (ctx.claim.value - lo) > 0.05:
        return Finding("sample_size", WARN, msg + " Too wide to trust.", data)
    return Finding("sample_size", OK, msg, data)


def per_class(ctx: Context) -> Finding:
    p = ctx.preds
    recalls = {}
    for cls, grp in p.groupby("y_true"):
        recalls[str(cls)] = float((grp["y_true"] == grp["y_pred"]).mean())
    balanced = float(np.mean(list(recalls.values())))
    acc = float((p["y_true"] == p["y_pred"]).mean())
    gap = acc - balanced
    data = {"balanced_accuracy": round(balanced, 4), "recall": {k: round(v, 3) for k, v in recalls.items()}}
    msg = f"Accuracy is {acc:.1%} but balanced accuracy is {balanced:.1%}."
    if gap > 0.30:
        return Finding("per_class", FAIL, msg + " Some classes are mostly wrong.", data)
    if gap > 0.10:
        return Finding("per_class", WARN, msg + " Uneven across classes.", data)
    return Finding("per_class", OK, msg, data)


def leakage(ctx: Context) -> Finding:
    test, train = ctx.preds, ctx.train
    cols = [c for c in test.columns if c not in ("y_true", "y_pred") and c in train.columns]
    if not cols:
        return Finding("leakage", NA, "No shared feature columns to compare.")
    h_test = pd.util.hash_pandas_object(test[cols].round(4), index=False)
    h_train = set(pd.util.hash_pandas_object(train[cols].round(4), index=False))
    overlap = float(h_test.isin(h_train).mean())
    data = {"overlap": round(overlap, 4), "columns": cols}
    msg = f"{overlap:.1%} of test rows also appear in the training data."
    if overlap > 0.05:
        return Finding("leakage", FAIL, msg + " Train/test leakage.", data)
    if overlap > 0.01:
        return Finding("leakage", WARN, msg, data)
    return Finding("leakage", OK, msg + " No meaningful overlap.", data)


_FAIL_PATTERNS = [
    r"(tun|select|chos|optimi[sz]|pick)\w*\W+(\w+\W+){0,8}(on|using|against)\s+the\s+test",
    r"no\s+(held-?out|separate|independent)\s+(test|validation)",
    r"test\s+set\W+(\w+\W+){0,3}(used|was used)\s+(for|to)\s+(tun|select|choos)",
]
_WARN_PATTERNS = [r"single\s+(random\s+)?split", r"no\s+cross-?validation",
                  r"not\s+(specified|reported)"]


def eval_method(ctx: Context) -> Finding:
    notes = ctx.claim.eval_notes
    for pat in _FAIL_PATTERNS:
        m = re.search(pat, notes, re.I)
        if m:
            return Finding("eval_method", FAIL,
                           f'Evaluation notes say: "{m.group(0).strip()}". The test set shaped the result.',
                           {"match": m.group(0)})
    for pat in _WARN_PATTERNS:
        m = re.search(pat, notes, re.I)
        if m:
            return Finding("eval_method", WARN,
                           f'Weak evaluation protocol: "{m.group(0).strip()}".', {"match": m.group(0)})
    return Finding("eval_method", OK, "No red flags in the described evaluation protocol.")


def paper_check(ctx: Context) -> Finding:
    text = ctx.claim.paper_text
    claimed = ctx.claim.value * 100
    reported = []
    for m in re.finditer(r"accuracy", text, re.I):
        window = text[max(0, m.start() - 120): m.end() + 120]
        reported += [float(x) for x in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", window)]
    reported = sorted(set(reported))
    has_baseline = bool(re.search(r"baseline|majority|compared (to|with)", text, re.I))
    data = {"reported": reported, "has_baseline": has_baseline}
    if not reported:
        return Finding("paper_check", WARN, "The paper never reports this metric as a percentage.", data)
    if not any(abs(r - claimed) <= 0.5 for r in reported):
        shown = ", ".join(f"{r:g}%" for r in reported)
        return Finding("paper_check", FAIL,
                       f"The paper reports {shown}, not the claimed {claimed:g}%.", data)
    if not has_baseline:
        return Finding("paper_check", WARN,
                       "Number matches the paper, but no baseline is given for context.", data)
    return Finding("paper_check", OK, "Paper reports the same number and gives baselines.", data)


# -------------------------------------------------------------- registry
@dataclass
class Tool:
    name: str
    label: str
    needs: str
    func: Callable[[Context], Finding]
    available: Callable[[Context], bool]


def _has_preds(c): return c.preds is not None
def _has_both(c): return c.preds is not None and c.train is not None
def _has_notes(c): return bool(c.claim.eval_notes.strip())
def _has_paper(c): return bool(c.claim.paper_text.strip())


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("class_balance", "Class balance", "predictions CSV", class_balance, _has_preds),
    Tool("recompute_metric", "Recompute metric", "predictions CSV", recompute_metric, _has_preds),
    Tool("sample_size", "Sample size", "predictions CSV", sample_size, _has_preds),
    Tool("per_class", "Per-class recall", "predictions CSV", per_class, _has_preds),
    Tool("leakage", "Train/test leakage", "predictions + train CSV", leakage, _has_both),
    Tool("eval_method", "Evaluation method", "evaluation notes", eval_method, _has_notes),
    Tool("paper_check", "Paper cross-check", "paper text", paper_check, _has_paper),
]}
