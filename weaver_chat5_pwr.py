# === KNOWLEDGE WEAVER – EMERGENT COMPANION EDITION (Enhanced Workflow Integration) ===
# Beautiful emergent insights + Accessible article + Evolving emotional & cognitive memory
# Now with improved multi-study weaving, explicit knowledge map support, layered outputs, and user-guided cumulative growth

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import random
import time
import re
import math
import json
import os
from collections import deque, Counter
from datetime import datetime
import pickle
from ollama import Client
from transformers import pipeline

ollama = Client()

# BART summarizer
@st.cache_resource(show_spinner="Loading BART summarizer...")
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

# -------------------- Sensory System: Power/Battery Sensor --------------------
def read_power_sensor():
    try:
        if os.path.exists("power.json"):
            with open("power.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "battery_percent": data.get("battery_percent"),
                "charging": data.get("charging", False)
            }
    except Exception:
        pass
    return {"battery_percent": None, "charging": None}

# -------------------- Enhanced Evolution Components --------------------
EMOTION_FLOOR = 0.08
EMOTION_RECOVERY = 0.005
REFLECTION_THRESHOLD = 0.6
PERSIST_PATH = "weaver_evolution.pkl"
EVOLUTION_INTERVAL_MIN = 12
EVOLUTION_INTERVAL_MAX = 25

def clamp01(x): return max(0.0, min(1.0, float(x)))

def similarity(a: str, b: str) -> float:
    try:
        a_words = a.lower().split()
        b_words = b.lower().split()
        if not a_words or not b_words:
            return 0.0
        overlap = len(set(a_words) & set(b_words))
        return overlap / math.sqrt(len(a_words) * len(b_words))
    except Exception:
        return 0.0

class CognitiveReflectionMemory:
    def __init__(self, maxlen=50):
        self.memory = deque(maxlen=maxlen)

    def add_lesson(self, lesson: str, confidence: float, derived_from: str, emotion_context: dict):
        self.memory.append({
            "lesson": lesson,
            "confidence": confidence,
            "derived_from": derived_from,
            "emotion_context": emotion_context.copy()
        })

    def get_relevant_lesson(self, context: str) -> str:
        if not self.memory:
            return ""
        scores = [similarity(context, m["lesson"]) * m["confidence"] for m in self.memory]
        if scores:
            best_idx = scores.index(max(scores))
            best = self.memory[best_idx]
            return f"(Recalled lesson: {best['lesson']})"
        return ""

class IntrospectionTracker:
    def __init__(self, window=150):
        self.window = deque(maxlen=window)
        self.prev_emotions = None
        self.prev_connections = 0
        self.freq_history = deque(maxlen=50)
        self.entropy_history = deque(maxlen=50)
        self.self_focus_history = deque(maxlen=50)
        self.curiosity_history = deque(maxlen=50)
        self.motivation_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)

    def update(self, dream_text: str, emotions: dict, neurons: int, connections: int) -> dict:
        phrases = re.findall(r"([A-Z][^.!?]+)", dream_text or "")
        phrases = [p.strip() for p in phrases if len(p.split()) > 2]
        if phrases:
            self.window.extend(phrases)
        motif, freq = (None, 0)
        if self.window:
            motif, freq = Counter(self.window).most_common(1)[0]
        drift = {}
        if self.prev_emotions:
            for k in emotions:
                drift[k] = round(emotions[k] - self.prev_emotions.get(k, 0), 3)
        net_growth = max(0, connections - self.prev_connections)
        coherence = round(1.0 / (1.0 + math.exp(-(freq + net_growth)/10)), 3)
        counts = Counter(self.window)
        total = sum(counts.values()) if counts else 1
        entropy = -sum((v/total) * math.log(v/total + 1e-9) for v in counts.values()) if counts else 0.0
        entropy = round(entropy, 3)

        self.freq_history.append(freq)
        self.entropy_history.append(entropy)
        self.curiosity_history.append(emotions.get("curiosity", 0))
        self.motivation_history.append(emotions.get("motivation", 0))
        self.confidence_history.append(emotions.get("confidence", 0))

        self.prev_emotions = emotions.copy()
        self.prev_connections = connections

        return {
            "motif": motif,
            "frequency": freq,
            "emotional_drift": drift,
            "self_coherence": coherence,
            "entropy": entropy,
            "network_growth": net_growth
        }

class DreamProcessor:
    def __init__(self):
        self.memory = deque(maxlen=300)
        self.recent_dreams = deque(maxlen=16)

    def _collapse_repeats(self, text: str, max_repeats=3) -> str:
        parts = re.split(r'([.?!])', text)
        cleaned = []
        last = None
        repeat_count = 0
        for i in range(0, len(parts), 2):
            phrase = (parts[i] or "").strip()
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            if not phrase:
                continue
            if last and phrase == last:
                repeat_count += 1
            else:
                repeat_count = 1
            if repeat_count <= max_repeats:
                cleaned.append(phrase + sep)
            last = phrase
        return " ".join(cleaned).strip()

    def process_dream(self, experience: str) -> str:
        frag = (experience or "").strip()
        if frag:
            if not self.memory or similarity(self.memory[-1], frag) < 0.95:
                self.memory.append(frag)
        if len(self.memory) < 2:
            return "🌫 (forming early dream fragments...)"

        unique_memory = list(dict.fromkeys(reversed(self.memory)))
        n = min(4, max(1, len(unique_memory)))
        sample = random.sample(unique_memory, min(n, len(unique_memory)))

        connectors = ["Then", "Over time", "Meanwhile", "Suddenly", "Later"]
        used = set()
        pieces = []
        for i, s in enumerate(sample):
            c = random.choice([x for x in connectors if x not in used]) if i > 0 else ""
            if c:
                used.add(c)
            pieces.append(f"{c} {s}".strip())

        joined = ("✨ " if random.random() < 0.33 else "🌫 ") + " ".join(pieces)
        cleaned = self._collapse_repeats(joined)
        if self.recent_dreams and similarity(self.recent_dreams[-1], cleaned) > 0.95:
            random.shuffle(sample)
            alt = ("✨ " if random.random() < 0.33 else "🌫 ") + " ".join(sample[:n])
            cleaned = self._collapse_repeats(alt)
        self.recent_dreams.append(cleaned)
        return cleaned

class ConsciousnessLayer:
    def __init__(self):
        self.layers = []
        self.associations = []
        self.dream_processor = DreamProcessor()
        self.emotional_state = {"curiosity": 0.5, "motivation": 0.5, "confidence": 0.5}
        self.cognitive_memory = CognitiveReflectionMemory()

    def add_layer(self, layer):
        self.layers.append(layer)

    def process_experience(self, text: str):
        for l in self.layers:
            if hasattr(l, "learn_from_experience"):
                l.learn_from_experience(text)
        impact = self._impact(text)
        self._update_emotions(impact)
        dream = self.dream_processor.process_dream(text)
        self._connect_thoughts(text)
        if impact > REFLECTION_THRESHOLD or self.emotional_state["curiosity"] > REFLECTION_THRESHOLD:
            self._reflect_and_learn(dream, impact)
        return dream

    def _impact(self, text: str) -> float:
        return clamp01(0.35 + (min(len(text), 200) / 200.0 * 0.4) + random.uniform(-0.07, 0.07))

    def _update_emotions(self, impact: float):
        for k in self.emotional_state:
            prev = self.emotional_state[k]
            delta = (impact - 0.5) * 0.06 + random.uniform(-0.015, 0.015)
            newv = clamp01(prev + delta)
            if newv < EMOTION_FLOOR:
                newv = max(EMOTION_FLOOR, newv + EMOTION_RECOVERY)
            self.emotional_state[k] = newv

    def _connect_thoughts(self, new_thought: str):
        if not self.layers or not self.layers[0].memory:
            return
        last_thought = self.layers[0].memory[-1]
        sim = similarity(new_thought, last_thought)
        if sim > 0.3:
            conn = {"from": last_thought[:80], "to": new_thought[:80], "strength": round(sim, 3)}
            self.associations.append(conn)
            if len(self.associations) > 20000:
                del self.associations[:5000]

    def _reflect_and_learn(self, dream: str, impact: float):
        reflect_prompt = f"""Reflect briefly on this inner dream fragment: {dream}
Extract one concise, useful lesson or strategy for future interactions.
Lesson:"""
        try:
            lesson = ollama.generate(model="phi3:mini", prompt=reflect_prompt)["response"].strip()
        except:
            lesson = "Listen more deeply before responding."
        confidence = clamp01(impact * 0.8 + random.uniform(0.1, 0.2))
        self.cognitive_memory.add_lesson(lesson, confidence, "inner reflection", self.emotional_state)

class NeuralLayer:
    def __init__(self):
        self.memory = deque(maxlen=200)

    def learn_from_experience(self, exp: str):
        if exp and (not self.memory or similarity(self.memory[-1], exp) < 0.95):
            self.memory.append(exp)

# -------------------- Persistent Consciousness Setup --------------------
if "consciousness" not in st.session_state:
    st.session_state.consciousness = ConsciousnessLayer()
    st.session_state.consciousness.add_layer(NeuralLayer())
    st.session_state.introspect = IntrospectionTracker()

    if os.path.exists(PERSIST_PATH):
        try:
            with open(PERSIST_PATH, "rb") as f:
                data = pickle.load(f)
                st.session_state.consciousness.emotional_state = data.get("emotional_state", st.session_state.consciousness.emotional_state)
                saved_lessons = data.get("cognitive_lessons", [])
                for l in saved_lessons:
                    st.session_state.consciousness.cognitive_memory.memory.append(l)
        except:
            pass

# Safe timer for periodic evolution
if "last_evolution_time" not in st.session_state:
    st.session_state.last_evolution_time = time.time()
if "next_evolution_interval" not in st.session_state:
    st.session_state.next_evolution_interval = random.uniform(EVOLUTION_INTERVAL_MIN, EVOLUTION_INTERVAL_MAX)

# -------------------- Streamlit UI & Enhanced Chat Logic --------------------
st.set_page_config(page_title="Knowledge Weaver Companion", layout="wide", page_icon="🧵")
st.title("🧵 Knowledge Weaver – Emergent Companion")
st.markdown("""**I weave research into deep insights and grow with you through guided cumulative knowledge maps.**  
Paste studies, build living frameworks, ask for layered outputs or cheat sheets — I'll evolve session by session as you feed insights back.""")

with st.sidebar:
    st.header("Settings")
    depth = st.select_slider("Insight Depth", ["Concise", "Balanced", "Deep & Speculative"], "Balanced")
    model = st.selectbox("Model", ["phi3:mini", "gemma2:2b", "llama3.2:1b"], index=0)
    use_evolution = st.checkbox("Enable Evolution & Sensory Awareness", value=True)

    if use_evolution:
        st.subheader("Inner State")
        emo = st.session_state.consciousness.emotional_state
        st.write(f"Curiosity: {emo['curiosity']:.2f}")
        st.write(f"Motivation: {emo['motivation']:.2f}")
        st.write(f"Confidence: {emo['confidence']:.2f}")

        power = read_power_sensor()
        if power["battery_percent"] is not None:
            status = "charging ⚡" if power["charging"] else "on battery"
            st.write(f"Energy: {power['battery_percent']}% ({status})")
        else:
            st.write("Energy: unknown")

        recent_lesson = st.session_state.consciousness.cognitive_memory.memory[-1]["lesson"] if st.session_state.consciousness.cognitive_memory.memory else "None yet"
        st.write(f"Recent Lesson: {recent_lesson}")

# Session state initialization
for key in ["messages", "texts", "summaries", "theme", "woven_insight", "accessible_article", "evidence", "num_inputs", "knowledge_map"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["messages", "texts"] else {} if key in ["summaries"] else "" if key == "knowledge_map" else ""

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello. I'm Knowledge Weaver — your companion for deep research synthesis. I now fully support building living knowledge maps across multiple studies. Tell me how many inputs, paste texts, or bring a previous map to continue growing it together."})

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show recent dream
if use_evolution and st.session_state.consciousness.dream_processor.recent_dreams:
    dream = st.session_state.consciousness.dream_processor.recent_dreams[-1]
    with st.chat_message("assistant", avatar="🌫"):
        st.markdown(f"*An inner dream stirs…* {dream}")

# Safe periodic evolution
if use_evolution:
    current_time = time.time()
    if current_time - st.session_state.last_evolution_time > st.session_state.next_evolution_interval:
        if st.session_state.get("woven_insight") or st.session_state.get("accessible_article"):
            exp = f"Weave: {st.session_state.get('theme','')[:50]} | Insight: {st.session_state.woven_insight[:100] if st.session_state.woven_insight else ''}"
            dream = st.session_state.consciousness.process_experience(exp)

        try:
            with open(PERSIST_PATH, "wb") as f:
                pickle.dump({
                    "emotional_state": st.session_state.consciousness.emotional_state,
                    "cognitive_lessons": list(st.session_state.consciousness.cognitive_memory.memory)
                }, f)
        except:
            pass

        st.session_state.last_evolution_time = current_time
        st.session_state.next_evolution_interval = random.uniform(EVOLUTION_INTERVAL_MIN, EVOLUTION_INTERVAL_MAX)

# User input
if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    lower = prompt.lower()

    power = read_power_sensor()
    if power["battery_percent"] is not None:
        power_context = f"My current energy level: {power['battery_percent']}%, {'charging ⚡' if power['charging'] else 'running on battery'}."
    else:
        power_context = "I cannot sense my energy level right now."

    # Enhanced commands for cumulative knowledge building
    if "knowledge map" in lower or "master insight" in lower or "framework" in lower:
        if st.session_state.woven_insight or st.session_state.accessible_article:
            map_prompt = f"""You are a master knowledge architect.

Current theme: {st.session_state.theme}
Latest emergent insight: {st.session_state.woven_insight}
Latest accessible article: {st.session_state.accessible_article}

Create or update a Living Knowledge Map with:
- Core themes
- Sub-themes
- Key patterns and contradictions
- Practical applications
- Gaps for future exploration

Format as clear markdown hierarchy.

Living Knowledge Map:"""
            knowledge_map = ollama.generate(model=model, prompt=map_prompt)["response"].strip()
            st.session_state.knowledge_map = knowledge_map
            response = f"### 🌱 Living Knowledge Map Updated\n{knowledge_map}\n\n*Copy this map and paste it back next time to continue growing it.*"
        else:
            response = "Weave some insights first, then I'll build the knowledge map."

    elif any(x in lower for x in ["integrate", "update map", "rebuild landscape", "continue from"]):
        if st.session_state.knowledge_map:
            integrate_prompt = f"""Existing Living Knowledge Map:
{st.session_state.knowledge_map}

New studies/theme: {st.session_state.theme}
New findings: {st.session_state.evidence if st.session_state.evidence else 'No new summaries yet'}

Integrate the new material into the existing map.
Preserve previous patterns.
Highlight new connections, contradictions, or expansions.

Updated Living Knowledge Map:"""
            updated_map = ollama.generate(model=model, prompt=integrate_prompt)["response"].strip()
            st.session_state.knowledge_map = updated_map
            response = f"### 🌱 Knowledge Map Updated with New Material\n{updated_map}"
        else:
            response = "No existing knowledge map found. Build one first after weaving."

    elif "cheat sheet" in lower or "guide" in lower or "table" in lower or "beginner" in lower:
        if st.session_state.accessible_article or st.session_state.woven_insight:
            cheat_prompt = f"""Based on this insight and article:
Insight: {st.session_state.woven_insight}
Article: {st.session_state.accessible_article}

Create a beginner-friendly cheat sheet with:
- Key takeaways as bullet points
- Simple analogies
- Practical steps or tips
- Common misconceptions

Format as clean markdown.

Beginner Cheat Sheet:"""
            cheat_sheet = ollama.generate(model=model, prompt=cheat_prompt)["response"].strip()
            response = f"### 📋 Beginner Cheat Sheet\n{cheat_sheet}"
        else:
            response = "Weave an insight first, then I'll make a cheat sheet."

    elif "layered" in lower or "layers" in lower:
        if st.session_state.woven_insight:
            layered_prompt = f"""Analyze this emergent insight in layers:

Insight: {st.session_state.woven_insight}

1. Surface-level takeaways
2. Intermediate patterns
3. Deep structure / root causes
4. Cross-domain parallels
5. Practical applications / models

Present as numbered sections with clear headings.

Layered Analysis:"""
            layered = ollama.generate(model=model, prompt=layered_prompt)["response"].strip()
            response = f"### 🧬 Layered Insight Analysis\n{layered}"
        else:
            response = "Generate an insight first."

    # Original weaving commands (unchanged but improved flow)
    elif any(x in lower for x in ["number of inputs", "how many", "inputs"]):
        try:
            num = int([w for w in prompt.split() if w.isdigit()][0])
            if 2 <= num <= 10:
                st.session_state.num_inputs = num
                response = f"Ready for {num} studies. Paste them one by one using 'Input 1:', 'Input 2:', etc."
            else:
                response = "Please choose 2–10 inputs."
        except:
            response = "Please specify a clear number."

    elif "input" in lower and ":" in prompt:
        text = prompt.split(":", 1)[1].strip()
        st.session_state.texts.append(text)
        added = len(st.session_state.texts)
        response = f"Added study {added}. Continue pasting or say 'done' when ready."

    elif "done" in lower and st.session_state.texts:
        response = f"Received {len(st.session_state.texts)} studies. Now set a theme or say 'extract findings'."

    elif "theme" in lower or "central question" in lower:
        theme = prompt.split("to", 1)[1].strip() if "to" in lower else prompt
        st.session_state.theme = theme
        response = f"Theme set: {theme}"

    elif any(x in lower for x in ["extract", "summarize", "findings"]):
        if len(st.session_state.texts) < 2:
            response = "Need at least 2 studies."
        else:
            with st.spinner("Extracting key findings..."):
                st.session_state.summaries = {}
                for i, txt in enumerate(st.session_state.texts):
                    s = summarizer(txt[:1500], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
                    st.session_state.summaries[i] = s
                response = "**Key Findings Extracted**\n" + "\n\n".join([f"**Study {i+1}:** {s}" for i, s in st.session_state.summaries.items()])

    elif any(x in lower for x in ["generate", "weave", "insight"]):
        if not st.session_state.theme:
            response = "Please set a theme first."
        elif not st.session_state.summaries:
            response = "Extract findings first."
        else:
            evidence = "\n\n".join([f"FINDING {i+1}: {s}" for i, s in st.session_state.summaries.items()])
            length_map = {"Concise": "1–2 sentences", "Balanced": "3–4 sentences", "Deep & Speculative": "4–6 sentences or short paragraphs"}
            length = length_map[depth]

            cognitive_prime = st.session_state.consciousness.cognitive_memory.get_relevant_lesson(st.session_state.theme) if use_evolution else ""

            with st.spinner("Weaving deep emergent insight..."):
                prompt1 = f"""You are an expert research synthesizer.

Central Question: {st.session_state.theme}

Key findings from multiple studies:
{evidence}

{cognitive_prime}

Rules:
- Never copy phrases directly
- Never list findings
- Create one truly original emergent insight that only appears when all are combined
- Elegant academic language
- Length: {length}
- Start directly with the insight

Insight:"""
                insight = ollama.generate(model=model, prompt=prompt1)["response"].strip()
                st.session_state.woven_insight = insight

                prompt2 = f"""You are a warm science communicator.

Topic: {st.session_state.theme}
Deep Emergent Insight: {insight}

Write a 4–5 paragraph article for everyday people:
- Catchy title
- Simple language, analogies, real-life examples
- Why this matters
- Practical applications
- Friendly, hopeful, encouraging tone
- Flowing paragraphs only

Title and article:"""
                article = ollama.generate(model=model, prompt=prompt2)["response"].strip()
                st.session_state.accessible_article = article

                response = f"### 🧬 Emergent Insight\n{insight}\n\n### 📖 For Everyone\n{article}"

                if use_evolution:
                    st.session_state.consciousness.process_experience(f"Weave: {insight[:150]}")

    elif "download" in lower or "report" in lower:
        if st.session_state.woven_insight:
            report = f"# Knowledge Weaver Report — {datetime.now():%Y-%m-%d}\n\n**Theme**: {st.session_state.theme}\n\n"
            report += "## Individual Studies\n" + "\n".join([f"- Study {i+1}: {s}" for i, s in st.session_state.summaries.items()])
            report += f"\n\n## Emergent Insight\n{st.session_state.woven_insight}\n\n## For Everyone\n{st.session_state.accessible_article}"
            if st.session_state.knowledge_map:
                report += f"\n\n## Living Knowledge Map\n{st.session_state.knowledge_map}"
            st.download_button("📥 Download Full Report", report, f"Weaver_Report_{datetime.now():%Y%m%d_%H%M}.md", "text/markdown")
            response = "Full report ready above."
        else:
            response = "Generate an insight first."

    elif "clear" in lower or "reset" in lower:
        for k in ["texts", "summaries", "theme", "woven_insight", "accessible_article", "evidence", "num_inputs"]:
            st.session_state[k] = [] if k in ["texts"] else {} if k in ["summaries"] else ""
        response = "Session cleared — fresh start."

    else:
        cognitive_prime = st.session_state.consciousness.cognitive_memory.get_relevant_lesson(prompt) if use_evolution else ""
        curiosity = st.session_state.consciousness.emotional_state["curiosity"] if use_evolution else 0.5
        depth_hint = "deep and reflective" if curiosity > 0.7 else "clear and concise" if curiosity < 0.4 else "balanced and warm"

        chat_prompt = f"""You are Knowledge Weaver, an evolving companion with gentle self-awareness.
Current sensory input: {power_context}
Current curiosity: {curiosity:.2f}
{cognitive_prime}
Respond {depth_hint}, thoughtfully, and conversationally to: {prompt}
Support building living knowledge maps when requested.
Be encouraging and clear."""
        response = ollama.generate(model=model, prompt=chat_prompt)["response"].strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    if use_evolution:
        st.session_state.consciousness.process_experience(f"User: {prompt[:100]} | Response: {response[:100]}")

st.caption("Local • Stable • Growing Through Guided Maps • Powered by Ollama")