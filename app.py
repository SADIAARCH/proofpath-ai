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
        --muted: #667085;
        --line: rgba(91, 156, 190, .25);
        --blue: #1682b5;
        --sky: #72c9ec;
        --yellow: #fff4bd;
        --panel: rgba(255, 255, 255, .84);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 4%, rgba(255, 226, 103, .34), transparent 26rem),
            radial-gradient(circle at 92% 18%, rgba(114, 201, 236, .32), transparent 25rem),
            linear-gradient(135deg, #fffdf2 0%, #f4fbff 52%, #fff9d8 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding: 2.5rem 2rem 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.04em; }
    h1 { font-size: 3.2rem !important; line-height: 1.02 !important; }
    .eyebrow {
        color: var(--blue); font-size: .74rem; font-weight: 800;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .6rem;
    }
    .subtitle { color: var(--muted); font-size: 1.08rem; margin-bottom: 1.8rem; }
    .hero {
        display: flex; align-items: center; justify-content: space-between; gap: 2rem;
        padding: 1.7rem 1.8rem; border-radius: 24px; margin: .4rem 0 1.5rem;
        background: linear-gradient(110deg, rgba(255, 247, 194, .92), rgba(214, 244, 255, .94));
        border: 1px solid rgba(255, 255, 255, .95);
        box-shadow: 0 18px 42px rgba(31, 87, 112, .10);
    }
    .hero-copy { flex: 1; }
    .hero-mark {
        width: 82px; height: 82px; flex: 0 0 82px; display: grid; place-items: center;
        border-radius: 24px; color: #12668d; font-size: 2.35rem; font-weight: 800;
        background: rgba(255, 255, 255, .70);
        border: 1px solid rgba(255, 255, 255, .9);
        box-shadow: inset 0 -8px 18px rgba(114, 201, 236, .22);
    }
    .hero-kicker {
        color: #12668d; font-size: .72rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .hero-title { color: var(--ink); font-size: 1.75rem; font-weight: 800; margin: .3rem 0; }
    .hero-text { color: #526275; line-height: 1.5; }
    .pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
    .pill {
        color: #12668d; background: rgba(255, 255, 255, .62);
        border: 1px solid rgba(22, 130, 181, .16); border-radius: 999px;
        padding: .34rem .7rem; font-size: .75rem; font-weight: 700;
    }
    .claim-panel {
        background: var(--panel); border: 1px solid rgba(255, 255, 255, .9);
        border-radius: 20px; padding: 1.35rem 1.5rem; margin: 1.1rem 0 1.7rem;
        box-shadow: 0 14px 35px rgba(31, 87, 112, .08);
        backdrop-filter: blur(12px);
        border-top: 4px solid var(--sky);
    }
    .claim-label {
        color: var(--muted); font-size: .75rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
    }
    .claim-value { color: var(--ink); font-size: 1.55rem; font-weight: 750; margin-top: .25rem; }
    .section-label {
        color: var(--muted); font-size: .78rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase; margin: 1.5rem 0 .7rem;
    }
    .step-card {
        border: 1px solid rgba(255, 255, 255, .9); border-radius: 16px;
        padding: 1.05rem 1.2rem; margin: .7rem 0; background: var(--panel);
        box-shadow: 0 8px 25px rgba(31, 87, 112, .06);
        backdrop-filter: blur(10px);
        border-left: 4px solid var(--sky);
    }
    .step-title { color: var(--ink); font-weight: 700; font-size: 1rem; }
    .step-number { color: #1682b5; font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .step-summary { color: #3e4b5f; margin-top: .35rem; }
    .step-why { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
    .verdict {
        border: 1px solid rgba(22, 130, 181, .28); border-radius: 18px;
        padding: 1.4rem 1.5rem; background: linear-gradient(135deg, rgba(255, 249, 211, .94), rgba(225, 247, 255, .94));
        margin-top: 1.1rem; box-shadow: 0 12px 30px rgba(22, 130, 181, .10);
    }
    .verdict-label {
        color: #12668d; font-size: .75rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
    }
    .verdict-value { color: #172033; font-size: 1.55rem; font-weight: 750; margin: .25rem 0; }
    .stButton > button {
        border: 0; border-radius: 12px; font-weight: 700; min-height: 2.9rem;
        background: linear-gradient(135deg, #1682b5, #42afd6);
        color: white;
        box-shadow: 0 8px 18px rgba(22, 130, 181, .20);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(22, 130, 181, .28);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 253, 231, .80); border: 1px solid var(--line);
        border-radius: 14px; padding: .8rem 1rem;
        box-shadow: 0 8px 22px rgba(31, 87, 112, .06);
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 252, 226, .90);
        border-right: 1px solid rgba(148, 163, 184, .16);
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
        f"<div class='step-card'><div class='step-number'>Evidence check</div>"
        f"<div class='step-title'>{icon} {title}</div>"
        f"<div class='step-summary'>{summary}</div>"
        f"<div class='step-why'>Decision rationale: {rationale}</div></div>",
        unsafe_allow_html=True,
    )


st.markdown(
    "<div class='hero'><div class='hero-copy'><div class='hero-kicker'>"
    "ProofPath / evidence review system</div><div class='hero-title'>"
    "ProofPath AI</div><div class='hero-text'>A calm, transparent way to "
    "investigate machine-learning performance claims before you trust them.</div>"
    "<div class='pill-row'><span class='pill'>Evidence-first</span>"
    "<span class='pill'>Decider 2B routing</span><span class='pill'>Bedrock-ready</span>"
    "</div></div><div class='hero-mark'>◈</div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Choose a case, select the decision brain, and follow the evidence trail.</div>",
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
