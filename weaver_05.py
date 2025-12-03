# === KNOWLEDGE WEAVER – FAST & POWERFUL FOR YOUR LAPTOP (Ollama + Phi-3 Mini) ===
# Beautiful emergent insights + Practical, real-world takeaways

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
for key in ["texts", "summaries", "woven_insight", "practical_takeaways"]:
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

        # === 1. Emergent Insight (academic style) ===
        prompt1 = f"""You are an expert research synthesizer.

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

        response1 = ollama.generate(model=model, prompt=prompt1)
        emergent_insight = response1['response'].strip()
        st.session_state.woven_insight = emergent_insight

        # === 2. Practical Takeaways (simple, real-world language) ===
        prompt2 = f"""You are a science communicator who makes research accessible to everyone.

Here is a deep insight from several studies:
{emergent_insight}

Central Question: {theme}

Explain this insight in simple, everyday language for non-experts (e.g., policymakers, teachers, community leaders, or the general public).
Focus on:
- What this means in real life
- Why it matters
- Possible actions or applications (e.g., conservation, health, education, policy)

Use bullet points.
Keep it friendly and encouraging.
Start directly with the takeaways (no introduction)."""

        response2 = ollama.generate(model=model, prompt=prompt2)
        practical = response2['response'].strip()
        st.session_state.practical_takeaways = practical

    st.success("Complete!")

    # Display Emergent Insight
    st.markdown("### 🧬 Emergent Insight")
    st.markdown(f"<div style='background:#f0f8ff;padding:28px;border-radius:12px;border-left:6px solid #4361ee;font-size:18px;line-height:1.8'>{emergent_insight}</div>", unsafe_allow_html=True)

    # Display Practical Takeaways
    st.markdown("### 🌱 Practical Takeaways (for everyone)")
    st.markdown(f"<div style='background:#e8f5e9;padding:25px;border-radius:12px;border-left:6px solid #4caf50;font-size:17px;line-height:1.7'>{practical}</div>", unsafe_allow_html=True)

# -------------------- Export --------------------
if st.session_state.woven_insight:
    report = f"# Knowledge Weaver Report — {datetime.now():%Y-%m-%d}\n\n**Central Question**: {theme}\n\n"
    report += "## Individual Findings\n"
    for i in st.session_state.summaries:
        report += f"- Input {i+1}: {st.session_state.summaries[i]}\n"
    report += f"\n## Emergent Insight\n{st.session_state.woven_insight}"
    if st.session_state.practical_takeaways:
        report += f"\n\n## Practical Takeaways\n{st.session_state.practical_takeaways}"

    st.download_button(
        "📥 Download Report",
        data=report,
        file_name=f"Weaver_Report_{datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown"
    )

st.caption("Fast & beautiful on Core i3 • Ollama + Phi-3 Mini • Academic + Real-world insights")