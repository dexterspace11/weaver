# === KNOWLEDGE WEAVER – FAST & POWERFUL FOR YOUR LAPTOP (Ollama + Phi-3 Mini) ===
# Beautiful emergent insights in seconds on Core i3

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from ollama import Client
from transformers import pipeline
from datetime import datetime

ollama = Client()  # connects to local Ollama

# BART for summaries (still works great)
@st.cache_resource(show_spinner="Loading BART summarizer...")
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

# -------------------- UI --------------------
st.set_page_config(page_title="Knowledge Weaver", layout="wide", page_icon="🧵")
st.title("🧵 Knowledge Weaver")
st.markdown("**Weave research into one deep, original insight – now fast on your laptop!**")

with st.sidebar:
    st.header("Settings")
    depth = st.select_slider("Insight Depth", ["Concise", "Balanced", "Deep & Speculative"], "Balanced")
    model = st.selectbox("Weaving model (Phi-3 Mini recommended)", ["phi3:mini", "gemma2:2b", "llama3.2:1b"], index=0)

# Session state
for key in ["texts", "summaries", "woven_insight"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "texts" else {} if key == "summaries" else None

# -------------------- Inputs --------------------
st.header("1. Input Studies")
num = st.number_input("Number of inputs", 2, 10, 3)
texts = []
for i in range(num):
    with st.expander(f"Input {i+1}", expanded=i < 2):
        t = st.text_area("Paste text", height=220, key=f"in{i}")
        if t.strip():
            texts.append(t.strip())

if len(texts) < 2:
    st.warning("Need at least 2 inputs")
    st.stop()

st.session_state.texts = texts

st.header("2. Central Question")
theme = st.text_input("What connects these studies?", placeholder="e.g., ethnobotanical knowledge in Assam forests")

# -------------------- Summaries --------------------
if st.button("Extract Key Findings", type="primary"):
    st.session_state.summaries = {}
    prog = st.progress(0)
    for i, text in enumerate(texts):
        summary = summarizer(text[:1500], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        st.session_state.summaries[i] = summary
        st.markdown(f"**Input {i+1}:** {summary}")
        prog.progress((i + 1) / len(texts))
    st.success("Summaries ready!")

# -------------------- Weaving --------------------
if st.button("🧵 Generate Woven Insight", type="primary"):
    if not theme.strip():
        st.warning("Please enter a central question")
        st.stop()

    with st.spinner(f"Weaving with {model} (10–30 seconds on your laptop)..."):
        evidence = ""
        for i in st.session_state.summaries:
            evidence += f"FINDING {i+1}: {st.session_state.summaries[i]}\n\n"

        length = {
            "Concise": "1–2 sentences",
            "Balanced": "3–4 sentences",
            "Deep & Speculative": "4–6 sentences or short paragraphs"
        }[depth]

        prompt = f"""You are an expert research synthesizer.

Central Question: {theme}

Key findings (combine ALL of them into one new insight):
{evidence}

Rules:
- Never copy phrases directly
- Never list or repeat the findings
- Create one original emergent insight that only appears when everything is combined
- Write in elegant academic language
- Length: {length}
- Start directly with the insight

Insight:"""

        response = ollama.generate(model=model, prompt=prompt)
        insight = response['response'].strip()
        st.session_state.woven_insight = insight

    st.success("Complete!")
    st.markdown("### 🧬 Emergent Insight")
    st.markdown(f"<div style='background:#f0f8ff;padding:28px;border-radius:12px;border-left:6px solid #4361ee;font-size:18px;line-height:1.8'>{insight}</div>", unsafe_allow_html=True)

# -------------------- Export --------------------
if st.session_state.woven_insight:
    report = f"# Knowledge Weaver Report — {datetime.now():%Y-%m-%d}\n\n**Central Question**: {theme}\n\n"
    report += "## Individual Findings\n"
    for i in st.session_state.summaries:
        report += f"- Input {i+1}: {st.session_state.summaries[i]}\n"
    report += f"\n## Emergent Insight\n{st.session_state.woven_insight}"

    st.download_button("📥 Download Report", data=report, file_name=f"Weaver_Report_{datetime.now():%Y%m%d_%H%M}.md", mime="text/markdown")

st.caption("Fast & beautiful on Core i3 • Ollama + Phi-3 Mini • Your private research tool")