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
        --blue: #5b3fd1;
        --sky: #9b7cff;
        --yellow: #f0eaff;
        --panel: rgba(255, 255, 255, .84);
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 4%, rgba(155, 124, 255, .26), transparent 26rem),
            radial-gradient(circle at 92% 18%, rgba(83, 198, 255, .24), transparent 25rem),
            linear-gradient(135deg, #f8f6ff 0%, #f3f4ff 52%, #faf7ff 100%);
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
        background: linear-gradient(110deg, rgba(235, 229, 255, .94), rgba(218, 242, 255, .94));
        border: 1px solid rgba(255, 255, 255, .95);
        box-shadow: 0 18px 42px rgba(31, 87, 112, .10);
    }
    .hero-copy { flex: 1; }
    .hero-mark {
        width: 112px; height: 112px; flex: 0 0 112px; display: grid; place-items: center;
        position: relative; border-radius: 50%; color: #5532c9; font-size: 2.35rem; font-weight: 800;
        background: rgba(255, 255, 255, .70);
        border: 1px solid rgba(255, 255, 255, .9);
        box-shadow: inset 0 -8px 18px rgba(155, 124, 255, .25);
    }
    .hero-mark::before, .hero-mark::after {
        content: ""; position: absolute; border: 1px solid rgba(91, 63, 209, .32);
        border-radius: 50%; animation: radar-pulse 3s ease-out infinite;
    }
    .hero-mark::before { inset: -12px; }
    .hero-mark::after { inset: -24px; animation-delay: 1.1s; }
    @keyframes radar-pulse {
        0% { opacity: .7; transform: scale(.86); }
        75%, 100% { opacity: 0; transform: scale(1.14); }
    }
    .hero-kicker {
        color: #5532c9; font-size: .72rem; font-weight: 800;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .hero-title { color: var(--ink); font-size: 1.75rem; font-weight: 800; margin: .3rem 0; }
    .hero-text { color: #526275; line-height: 1.5; }
    .pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }
    .pill {
        color: #5532c9; background: rgba(255, 255, 255, .68);
        border: 1px solid rgba(91, 63, 209, .18); border-radius: 999px;
        padding: .34rem .7rem; font-size: .75rem; font-weight: 700;
    }
    .flow {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem;
        margin: 0 0 1.7rem;
    }
    .flow-item {
        position: relative; padding: .9rem 1rem; border-radius: 14px;
        background: rgba(255, 255, 255, .66); border: 1px solid rgba(255, 255, 255, .9);
        box-shadow: 0 7px 18px rgba(31, 87, 112, .05);
    }
    .flow-item:not(:last-child)::after {
        content: "→"; position: absolute; right: -.62rem; top: .8rem;
        color: #5b3fd1; font-size: 1.15rem; font-weight: 800; z-index: 2;
    }
    .flow-kicker { color: #5b3fd1; font-size: .68rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
    .flow-title { color: var(--ink); font-size: .92rem; font-weight: 750; margin-top: .2rem; }
    .flow-copy { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
    .signal {
        display: inline-flex; align-items: center; gap: .4rem; color: #167b9f;
        background: rgba(255, 255, 255, .58); border: 1px solid rgba(22, 130, 181, .17);
        border-radius: 999px; padding: .3rem .65rem; font-size: .7rem; font-weight: 800;
        letter-spacing: .06em; text-transform: uppercase;
    }
    .signal-dot {
        width: .45rem; height: .45rem; border-radius: 50%; background: #8c64ff;
        box-shadow: 0 0 0 4px rgba(140, 100, 255, .16); animation: signal-blink 1.6s ease-in-out infinite;
    }
    @keyframes signal-blink { 50% { opacity: .35; transform: scale(.78); } }
    .workspace-card {
        min-height: 180px; padding: 1.25rem; border-radius: 18px;
        background: rgba(255, 255, 255, .72); border: 1px solid rgba(255, 255, 255, .92);
        box-shadow: 0 10px 26px rgba(54, 45, 118, .07);
    }
    .workspace-card h3 { margin: 0 0 .8rem; font-size: 1rem; letter-spacing: -.02em; }
    .workspace-label {
        color: #667085; font-size: .68rem; font-weight: 800;
        letter-spacing: .1em; text-transform: uppercase; margin-top: .8rem;
    }
    .workspace-value { color: #242044; font-size: .95rem; font-weight: 750; margin-top: .2rem; }
    .workspace-note { color: #667085; font-size: .8rem; line-height: 1.45; margin-top: .8rem; }
    .sidebar-brand {
        padding: .75rem .8rem; margin-bottom: 1rem; border-radius: 14px;
        background: linear-gradient(135deg, #eee8ff, #d9f5ff);
        border: 1px solid rgba(255, 255, 255, .9);
    }
    .sidebar-brand-title { color: #5532c9; font-size: 1rem; font-weight: 800; }
    .sidebar-brand-copy { color: #667085; font-size: .74rem; margin-top: .15rem; }
    @media (max-width: 720px) {
        h1 { font-size: 2.45rem !important; }
        .hero { padding: 1.25rem; }
        .hero-mark { width: 72px; height: 72px; flex-basis: 72px; font-size: 1.8rem; }
        .flow { grid-template-columns: 1fr; }
        .flow-item:not(:last-child)::after { content: "↓"; right: 1rem; top: auto; bottom: -.65rem; }
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
    .step-number { color: #5b3fd1; font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .step-summary { color: #3e4b5f; margin-top: .35rem; }
    .step-why { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
    .verdict {
        border: 1px solid rgba(22, 130, 181, .28); border-radius: 18px;
        padding: 1.4rem 1.5rem; background: linear-gradient(135deg, rgba(239, 234, 255, .96), rgba(222, 244, 255, .96));
        margin-top: 1.1rem; box-shadow: 0 12px 30px rgba(22, 130, 181, .10);
    }
    .verdict-label {
        color: #5532c9; font-size: .75rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
    }
    .verdict-value { color: #172033; font-size: 1.55rem; font-weight: 750; margin: .25rem 0; }
    .stButton > button {
        border: 0; border-radius: 12px; font-weight: 700; min-height: 2.9rem;
        background: linear-gradient(135deg, #5532c9, #8f6bff);
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
        background: rgba(244, 242, 255, .92);
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
st.markdown(
    "<div class='signal'><span class='signal-dot'></span> Evidence engine ready · deterministic checks online</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='flow'><div class='flow-item'><div class='flow-kicker'>01 · Context</div>"
    "<div class='flow-title'>Bring a claim</div><div class='flow-copy'>A result and its supporting artifacts.</div></div>"
    "<div class='flow-item'><div class='flow-kicker'>02 · Reason</div>"
    "<div class='flow-title'>Choose the next check</div><div class='flow-copy'>Decider 2B routes the evidence review.</div></div>"
    "<div class='flow-item'><div class='flow-kicker'>03 · Trust</div>"
    "<div class='flow-title'>Get a verdict</div><div class='flow-copy'>Deterministic tools show why.</div></div></div>",
    unsafe_allow_html=True,
)

cases_dir = Path(__file__).parent / "data" / "cases"
metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(cases_dir.glob("*/meta.json"))}

with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'><div class='sidebar-brand-title'>◈ ProofPath AI</div>"
        "<div class='sidebar-brand-copy'>Evidence before confidence.</div></div>",
        unsafe_allow_html=True,
    )
    theme = st.radio("Appearance", ["Day", "Night"], horizontal=True)
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

if theme == "Night":
    st.markdown(
        """
        <style>
        :root {
            --ink: #e8f4f8; --muted: #a6bdc8; --line: rgba(114, 201, 236, .24);
            --blue: #c1a9ff; --sky: #9b7cff; --panel: rgba(17, 20, 48, .80);
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 4%, rgba(155, 124, 255, .18), transparent 26rem),
                radial-gradient(circle at 92% 18%, rgba(83, 198, 255, .16), transparent 25rem),
                linear-gradient(135deg, #090b22 0%, #12143b 52%, #171343 100%);
            background-size: 140% 140%;
            animation: night-drift 18s ease-in-out infinite alternate;
        }
        @keyframes night-drift { from { background-position: 0% 0%; } to { background-position: 100% 100%; } }
        .hero {
            background: linear-gradient(110deg, rgba(44, 34, 86, .92), rgba(13, 55, 87, .92));
            border-color: rgba(114, 201, 236, .22);
            box-shadow: 0 18px 42px rgba(0, 0, 0, .28);
        }
        .hero-title, .claim-value, .step-title, .verdict-value, h1, h2, h3 { color: var(--ink); }
        .hero-text, .step-summary { color: #c0d3da; }
        .hero-mark { color: #c1a9ff; background: rgba(16, 17, 49, .72); border-color: rgba(193, 169, 255, .24); }
        .hero-mark::before, .hero-mark::after { border-color: rgba(155, 124, 255, .38); }
        .hero-kicker, .flow-kicker, .sidebar-brand-title, .verdict-label { color: #c1a9ff; }
        .pill, .flow-item, [data-testid="stMetric"] {
            color: #bcecf8; background: rgba(8, 31, 43, .62);
            border-color: rgba(114, 201, 236, .22);
        }
        .flow-title { color: var(--ink); }
        .flow-copy, .claim-label, .section-label, .step-why { color: #a6bdc8; }
        .workspace-card { background: rgba(14, 19, 54, .78); border-color: rgba(155, 124, 255, .2); }
        .workspace-label, .workspace-note { color: #a6bdc8; }
        .workspace-value, .workspace-card h3 { color: #e8e9ff; }
        .signal { color: #bcecf8; background: rgba(8, 31, 43, .62); border-color: rgba(114, 201, 236, .22); }
        .claim-panel, .step-card {
            background: rgba(14, 42, 56, .74); border-color: rgba(114, 201, 236, .18);
            box-shadow: 0 12px 30px rgba(0, 0, 0, .2);
        }
        .step-card { border-left-color: #58c5eb; }
        .verdict {
            background: linear-gradient(135deg, rgba(44, 34, 86, .92), rgba(12, 48, 82, .92));
            border-color: rgba(155, 124, 255, .38) !important;
        }
        [data-testid="stSidebar"] { background: rgba(6, 24, 34, .92); border-right-color: rgba(114, 201, 236, .18); }
        .sidebar-brand { background: linear-gradient(135deg, rgba(44, 34, 86, .94), rgba(11, 55, 84, .94)); border-color: rgba(155, 124, 255, .24); }
        </style>
        """,
        unsafe_allow_html=True,
    )

if source == "Demo case":
    if not metas:
        st.error("Run `python benchmark/make_cases.py` first.")
        st.stop()
    name = st.sidebar.selectbox("Case", list(metas))
    claim = Claim.from_case(metas[name], Path(__file__).parent)
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

setup_col, claim_col = st.columns([0.85, 1.55], gap="large")
with setup_col:
    st.markdown(
        f"<div class='workspace-card'><h3>Review workspace</h3>"
        f"<div class='workspace-label'>Input source</div>"
        f"<div class='workspace-value'>{source}</div>"
        f"<div class='workspace-label'>Decision brain</div>"
        f"<div class='workspace-value'>{strategy}</div>"
        f"<div class='workspace-note'>The router chooses the next evidence check. "
        "Deterministic tools calculate the finding.</div></div>",
        unsafe_allow_html=True,
    )
with claim_col:
    st.markdown(
        f"<div class='claim-panel' style='margin-top:0'><div class='claim-label'>"
        f"Claim under review</div><div class='claim-value'>{claim.text}</div></div>",
        unsafe_allow_html=True,
    )

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
