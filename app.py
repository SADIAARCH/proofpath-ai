"""ProofPath AI demo UI.  Run:  streamlit run app.py"""
import json
import tempfile
from pathlib import Path

import streamlit as st

from proofpath import (AdaptiveDecider, BedrockDecider, Claim, FixedOrderDecider,
                       ModelDecider, StrandsBedrockDecider, StrandsDecider,
                       run_agent)
from proofpath.tools import TOOLS

st.set_page_config(page_title="ProofPath AI", page_icon="🔎", layout="wide")

COLORS = {"ok": ("#16a34a", "✅"), "warn": ("#d97706", "⚠️"),
          "fail": ("#dc2626", "❌"), "unavailable": ("#6b7280", "➖")}
VERDICT = {"supported": ("#16a34a", "SUPPORTED"),
           "contradicted": ("#dc2626", "CONTRADICTED"),
           "insufficient_evidence": ("#d97706", "NOT ENOUGH EVIDENCE")}


def card(color, big, small=""):
    st.markdown(
        f"<div style='border-left:10px solid {color};padding:14px 18px;margin:10px 0;"
        f"border-radius:8px;background:{color}18'>"
        f"<div style='font-size:1.4rem;font-weight:700'>{big}</div>"
        f"<div style='font-size:1rem;opacity:.85'>{small}</div></div>",
        unsafe_allow_html=True)


st.title("🔎 ProofPath AI")
st.caption("The Evidence Detective: it decides what to check next before it judges a claim.")

cases_dir = Path(__file__).parent / "data" / "cases"
metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(cases_dir.glob("*/meta.json"))}

with st.sidebar:
    source = st.radio("Input", ["Demo case", "My own files"])
    strategy = st.selectbox(
        "Decision brain",
        ["Strands Decider 2B + Bedrock", "Amazon Bedrock (Nova)",
         "Adaptive (heuristic)", "Fixed checklist", "Decider 2B"],
    )
    if strategy == "Strands Decider 2B + Bedrock":
        st.caption("Decider 2B routes the next check; Bedrock explains findings.")
    elif strategy == "Amazon Bedrock (Nova)":
        st.caption("Bedrock routes the next check; Python tools own the verdict.")

if source == "Demo case":
    if not metas:
        st.error("Run `python benchmark/make_cases.py` first.")
        st.stop()
    name = st.sidebar.selectbox("Case", list(metas))
    claim = Claim.from_case(metas[name], Path(__file__).parent)
    st.write(f"**Claim:** {claim.text}")
else:
    value = st.sidebar.slider("Claimed accuracy", 0.50, 1.00, 0.98, 0.001)
    up_preds = st.sidebar.file_uploader("Predictions CSV (features, y_true, y_pred)", type="csv")
    up_train = st.sidebar.file_uploader("Training CSV (features, y_true)", type="csv")
    notes = st.sidebar.text_area("How was it evaluated?")
    paper = st.sidebar.text_area("Paper excerpt")
    tmp = Path(tempfile.mkdtemp())
    paths = {}
    for key, up in (("preds", up_preds), ("train", up_train)):
        if up:
            paths[key] = str(tmp / f"{key}.csv")
            Path(paths[key]).write_bytes(up.getvalue())
    claim = Claim(value=value, text=f"{value:.1%} accuracy", eval_notes=notes, paper_text=paper,
                  predictions_path=paths.get("preds"), train_path=paths.get("train"))

if st.button("Investigate", type="primary"):
    decider = {
        "Strands Decider 2B + Bedrock": StrandsBedrockDecider,
        "Amazon Bedrock (Nova)": BedrockDecider,
        "Adaptive (heuristic)": AdaptiveDecider,
        "Fixed checklist": FixedOrderDecider,
        "Decider 2B": ModelDecider,
    }[strategy]()
    report = run_agent(claim, decider)
    for s in report.steps:
        color, icon = COLORS[s.finding.status]
        card(color, f"{icon} Step {s.index}: {TOOLS[s.action].label}", s.finding.summary)
        st.caption(f"Why this check: {s.rationale}")
    color, label = VERDICT[report.verdict]
    card(color, f"Verdict: {label}", report.headline)
    c1, c2 = st.columns(2)
    c1.metric("Checks run", f"{len(report.steps)} of {len(TOOLS)}")
    c2.metric("Evidence coverage", f"{report.coverage:.0%}")
    if report.unverified:
        st.info("Could not verify (missing input): " + "; ".join(report.unverified))
