"""ProofPath AI demo UI. Run: python -m streamlit run app.py"""
import json
import tempfile
from pathlib import Path

import streamlit as st

from proofpath import (AdaptiveDecider, BedrockDecider, Claim, FixedOrderDecider,
                       ModelDecider, StrandsBedrockDecider, StrandsDecider,
                       run_agent)
from proofpath.tools import TOOLS

st.set_page_config(page_title="ProofPath AI | Evidence Review", page_icon="◈", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #637083;
        --line: #dce3ec;
        --blue: #1d4ed8;
        --panel: #f7f9fc;
    }
    .stApp { background: #ffffff; color: var(--ink); }
    [data-testid="stHeader"] { background: #ffffff; }
    .block-container { max-width: 1180px; padding-top: 2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    .eyebrow {
        color: var(--blue); font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.5rem; }
    .claim-panel {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 12px; padding: 1.25rem 1.4rem; margin: 1rem 0 1.5rem;
    }
    .claim-label {
        color: var(--muted); font-size: .75rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
    }
    .claim-value { color: var(--ink); font-size: 1.45rem; font-weight: 700; margin-top: .25rem; }
    .section-label {
        color: var(--muted); font-size: .78rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase; margin: 1.5rem 0 .7rem;
    }
    .step-card {
        border: 1px solid var(--line); border-radius: 10px;
        padding: 1rem 1.1rem; margin: .65rem 0; background: #fff;
    }
    .step-title { color: var(--ink); font-weight: 700; font-size: 1rem; }
    .step-summary { color: #3e4b5f; margin-top: .35rem; }
    .step-why { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
    .verdict {
        border: 2px solid #1d4ed8; border-radius: 12px;
        padding: 1.25rem 1.4rem; background: #eff6ff; margin-top: 1rem;
    }
    .verdict-label {
        color: #1e40af; font-size: .75rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
    }
    .verdict-value { color: #172033; font-size: 1.55rem; font-weight: 750; margin: .25rem 0; }
    .stButton > button {
        border-radius: 8px; font-weight: 650; min-height: 2.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {"ok": ("#16a34a", "✅"), "warn": ("#d97706", "⚠️"),
          "fail": ("#dc2626", "❌"), "unavailable": ("#6b7280", "➖")}
VERDICT = {"supported": ("#16a34a", "SUPPORTED"),
           "contradicted": ("#dc2626", "CONTRADICTED"),
           "insufficient_evidence": ("#d97706", "NOT ENOUGH EVIDENCE")}


def step_card(icon, title, summary, rationale):
    st.markdown(
        f"<div class='step-card'><div class='step-title'>{icon} {title}</div>"
        f"<div class='step-summary'>{summary}</div>"
        f"<div class='step-why'>Decision rationale: {rationale}</div></div>",
        unsafe_allow_html=True,
    )


st.markdown("<div class='eyebrow'>Evidence review system · v1.0</div>", unsafe_allow_html=True)
st.title("ProofPath AI")
st.markdown(
    "<div class='subtitle'>A transparent review agent for machine-learning performance claims.</div>",
    unsafe_allow_html=True,
)

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
    st.markdown(
        f"<div class='claim-panel'><div class='claim-label'>Claim under review</div>"
        f"<div class='claim-value'>{claim.text}</div></div>",
        unsafe_allow_html=True,
    )
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

st.markdown("<div class='section-label'>Review controls</div>", unsafe_allow_html=True)
if st.button("Run evidence review", type="primary", use_container_width=True):
    decider = {
        "Strands Decider 2B + Bedrock": StrandsBedrockDecider,
        "Amazon Bedrock (Nova)": BedrockDecider,
        "Adaptive (heuristic)": AdaptiveDecider,
        "Fixed checklist": FixedOrderDecider,
        "Decider 2B": ModelDecider,
    }[strategy]()
    report = run_agent(claim, decider)
    st.markdown("<div class='section-label'>Evidence trail</div>", unsafe_allow_html=True)
    for s in report.steps:
        _, icon = COLORS[s.finding.status]
        step_card(icon, f"Step {s.index} · {TOOLS[s.action].label}",
                  s.finding.summary, s.rationale)
    color, label = VERDICT[report.verdict]
    st.markdown(
        f"<div class='verdict' style='border-color:{color}'>"
        f"<div class='verdict-label'>Final assessment</div>"
        f"<div class='verdict-value'>{label}</div>"
        f"<div>{report.headline}</div></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Checks completed", f"{len(report.steps)} / {len(TOOLS)}")
    c2.metric("Evidence coverage", f"{report.coverage:.0%}")
    if report.unverified:
        st.warning("Unverified inputs: " + "; ".join(report.unverified))
