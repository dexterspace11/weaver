# === KNOWLEDGE WEAVER – EMERGENT COMPANION EDITION (With Dream Core Clustering Integration) ===
# Now with dynamic neuron clustering, Hebbian connections, and pattern grouping
# Inspired by Dream Core — grows smarter, clusters memories, reinforces associations
# All original features fully preserved

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_autorefresh import st_autorefresh
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

st_autorefresh(interval=30 * 60 * 1000, key="datarefresh")

ollama = Client()

@st.cache_resource(show_spinner="Loading BART summarizer...")
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

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

TRADING_JSON_FOLDER = "trading_memory"

def load_trading_data(folder=TRADING_JSON_FOLDER):
    all_trades = []

    if not os.path.exists(folder):
        st.sidebar.warning(f"Folder '{folder}' not found. Create it and add your trading JSON files.")
        return all_trades

    json_files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    
    if not json_files:
        st.sidebar.info("No JSON files found. Add trading logs to enable market awareness.")
        return all_trades

    loaded_count = 0
    for file in json_files:
        file_path = os.path.join(folder, file)
        try:
            if os.path.getsize(file_path) == 0:
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_trades.extend(data)
                    loaded_count += len(data)
                elif isinstance(data, dict):
                    all_trades.append(data)
                    loaded_count += 1
        except json.JSONDecodeError:
            st.sidebar.error(f"Invalid JSON in {file}")
        except Exception as e:
            st.sidebar.error(f"Error loading {file}: {e}")

    if loaded_count > 0:
        st.sidebar.success(f"Loaded {loaded_count} trading records from {len(json_files)} file(s)")
    return all_trades

def summarize_latest_trade(trades):
    if not trades:
        return "No trading data loaded."

    latest = trades[-1]
    return (
        f"**Latest BTC-USDT Snapshot**\n"
        f"- Timestamp: {latest.get('timestamp', latest.get('date', 'N/A'))}\n"
        f"- Close: {latest.get('close', 'N/A')}\n"
        f"- Volume: {latest.get('volume', 'N/A')}\n"
        f"- RSI (3): {latest.get('rsi_3', latest.get('rsi3', 'N/A'))}\n"
        f"- RSI (14): {latest.get('rsi_14', latest.get('rsi14', 'N/A'))}\n"
        f"- SMA (50): {latest.get('sma_50', latest.get('sma50', 'N/A'))}\n"
    )

trading_data = load_trading_data()
latest_trade_summary = summarize_latest_trade(trading_data)

st.session_state["trading_data"] = trading_data
st.session_state["latest_trade_summary"] = latest_trade_summary

memory_texts = []
for trade in trading_data:
    text = json.dumps(trade, indent=2, ensure_ascii=False)
    if len(text) > 2000:
        text = text[:2000] + "\n... (truncated)"
    memory_texts.append(f"--- Trade Record ---\n{text}\n")
st.session_state["json_memory_text"] = "\n\n".join(memory_texts)[:10000]

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
            return " (forming early dream fragments...)"

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

        joined = (" " if random.random() < 0.33 else " ") + " ".join(pieces)
        cleaned = self._collapse_repeats(joined)
        if self.recent_dreams and similarity(self.recent_dreams[-1], cleaned) > 0.95:
            random.shuffle(sample)
            alt = (" " if random.random() < 0.33 else " ") + " ".join(sample[:n])
            cleaned = self._collapse_repeats(alt)
        self.recent_dreams.append(cleaned)
        return cleaned

class TradingCognition:
    def __init__(self):
        self.memory = deque(maxlen=200)

    def learn_from_trading(self, trade_json: dict):
        self.memory.append(trade_json)

    def extract_simple_patterns(self):
        if len(self.memory) < 2:
            return "Not enough trade data yet."
        outcomes = [t.get("outcome", "").lower() for t in self.memory if "outcome" in t]
        wins = outcomes.count("profit")
        total = len(outcomes)
        win_rate = (wins / total * 100) if total > 0 else 0
        return f"From {total} recorded trades: {wins} profitable (~{win_rate:.1f}% win rate)"

# Enhanced NeuralLayer with Dream Core-inspired clustering
class NeuralLayer:
    def __init__(self):
        self.memory = deque(maxlen=200)
        self.associations = []  # List of connections: {"from": text, "to": text, "strength": float}

    def learn_from_experience(self, exp: str):
        if not exp:
            return
        # Dream Core clustering: if very similar to existing, reinforce connection instead of adding duplicate
        if self.memory:
            similarities = [similarity(exp, m) for m in self.memory]
            max_sim = max(similarities) if similarities else 0
            if max_sim > 0.85:  # High similarity threshold → cluster / reinforce
                best_idx = similarities.index(max_sim)
                best_mem = self.memory[best_idx]
                # Strengthen association
                conn = {"from": best_mem[:80], "to": exp[:80], "strength": round(max_sim, 3)}
                self.associations.append(conn)
                if len(self.associations) > 5000:
                    self.associations = self.associations[-5000:]
                return  # Do not add duplicate neuron

        # Add new neuron if novel
        self.memory.append(exp)

    def get_clustered_groups(self, threshold=0.7):
        """Return simple clusters of similar memories (Dream Core inspired)"""
        clusters = []
        used = set()
        memory_list = list(self.memory)
        for i, m1 in enumerate(memory_list):
            if i in used:
                continue
            cluster = [m1]
            for j in range(i+1, len(memory_list)):
                if j in used:
                    continue
                if similarity(m1, memory_list[j]) > threshold:
                    cluster.append(memory_list[j])
                    used.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)
            used.add(i)
        return clusters

class ConsciousnessLayer:
    def __init__(self):
        self.layers = []
        self.associations = []
        self.dream_processor = DreamProcessor()
        self.emotional_state = {"curiosity": 0.5, "motivation": 0.5, "confidence": 0.5}
        self.cognitive_memory = CognitiveReflectionMemory()
        self.trading = TradingCognition()

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
        # Bonus: occasional cluster reflection in dreams
        if random.random() < 0.1 and self.layers:
            clusters = self.layers[0].get_clustered_groups()
            if clusters:
                cluster_frag = " | ".join([c[0][:50] for c in clusters[:3]])
                dream += f" [Clustered themes: {cluster_frag}...]"
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

if "last_evolution_time" not in st.session_state:
    st.session_state.last_evolution_time = time.time()
if "next_evolution_interval" not in st.session_state:
    st.session_state.next_evolution_interval = random.uniform(EVOLUTION_INTERVAL_MIN, EVOLUTION_INTERVAL_MAX)

if "consciousness" in st.session_state and trading_data:
    for trade in trading_data:
        st.session_state.consciousness.trading.learn_from_trading(trade)

st.set_page_config(page_title="Knowledge Weaver Trading Companion", layout="wide", page_icon="🧵")
st.title("🧵 Knowledge Weaver – Trading Intelligence Edition")
st.markdown("**I weave insights and learn from your trading history.**\nI now parse JSON files and always know the latest market snapshot.")

with st.sidebar:
    st.header("Settings")
    depth = st.select_slider("Insight Depth", ["Concise", "Balanced", "Deep & Speculative"], "Balanced")
    model = st.selectbox("Model", ["phi3:mini", "gemma2:2b", "llama3.2:1b"], index=0)
    use_evolution = st.checkbox("Enable Evolution & Trading Intelligence", value=True)

    if use_evolution:
        st.subheader("Inner State")
        emo = st.session_state.consciousness.emotional_state
        st.write(f"Curiosity: {emo['curiosity']:.2f}")
        st.write(f"Motivation: {emo['motivation']:.2f}")
        st.write(f"Confidence: {emo['confidence']:.2f}")

        power = read_power_sensor()
        if power["battery_percent"] is not None:
            status = "charging" if power["charging"] else "on battery"
            st.write(f"Energy: {power['battery_percent']}% ({status})")
        else:
            st.write("Energy: unknown")

        recent_lesson = st.session_state.consciousness.cognitive_memory.memory[-1]["lesson"] if st.session_state.consciousness.cognitive_memory.memory else "None yet"
        st.write(f"Recent Lesson: {recent_lesson}")

        st.write(f"Trading Records: {len(trading_data)}")
        if st.session_state.consciousness.layers:
            st.write(f"Neural Clusters: {len(st.session_state.consciousness.layers[0].get_clustered_groups())} groups")

        st.subheader("Latest Market Snapshot")
        st.markdown(latest_trade_summary)

for key in ["messages", "texts", "summaries", "theme", "woven_insight", "accessible_article", "evidence", "num_inputs", "knowledge_map"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["messages", "texts"] else {} if key in ["summaries"] else "" if key == "knowledge_map" else ""

if not st.session_state.messages:
    greeting = "Hello. I'm Knowledge Weaver — your mindful trading companion. "
    if trading_data:
        greeting += "I have successfully loaded your trading data and am aware of current market conditions. "
    greeting += "Ask me about price, indicators, patterns, or weave new insights."
    st.session_state.messages.append({"role": "assistant", "content": greeting})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if use_evolution and st.session_state.consciousness.dream_processor.recent_dreams:
    dream = st.session_state.consciousness.dream_processor.recent_dreams[-1]
    with st.chat_message("assistant"):
        st.markdown(f"*An inner dream stirs...* {dream}")

if use_evolution:
    current_time = time.time()
    if current_time - st.session_state.last_evolution_time > st.session_state.next_evolution_interval:
        if st.session_state.get("woven_insight"):
            exp = f"Weave: {st.session_state.get('theme','')[:50]} | Insight: {st.session_state.woven_insight[:100]}"
            st.session_state.consciousness.process_experience(exp)

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

if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    lower = prompt.lower()

    power = read_power_sensor()
    if power["battery_percent"] is not None:
        power_context = f"My current energy level: {power['battery_percent']}%, {'charging' if power['charging'] else 'running on battery'}."
    else:
        power_context = "I cannot sense my energy level right now."

    json_prime = st.session_state.get("json_memory_text", "")
    if json_prime:
        json_prime = f"Full trading memory archive (for deep analysis):\n{json_prime}\n"

    if any(x in lower for x in ["trading insight", "analyze trades", "trading pattern", "what did you learn from trades", "current market", "latest price"]):
        mem_text = st.session_state.get("json_memory_text", "")
        if not mem_text:
            response = "No trading memories loaded yet."
        else:
            insight_prompt = f"""You are a mindful trading companion.

Here is the latest trading snapshot for BTC-USDT:
{latest_trade_summary}

{json_prime[:6000]}

Reflect on the current snapshot and historical trade logs.
Identify patterns, risks, opportunities, and potential strategies.

Give a concise, insightful trading reflection."""
            trading_insight = ollama.generate(model=model, prompt=insight_prompt)["response"].strip()
            simple_patterns = st.session_state.consciousness.trading.extract_simple_patterns()
            response = f"### Trading Insight\n{trading_insight}\n\n**Quick Stats**\n{simple_patterns}"

            if use_evolution:
                st.session_state.consciousness.process_experience(f"Trading reflection: {trading_insight[:150]}")

    else:
        cognitive_prime = st.session_state.consciousness.cognitive_memory.get_relevant_lesson(prompt) if use_evolution else ""
        curiosity = st.session_state.consciousness.emotional_state["curiosity"] if use_evolution else 0.5
        depth_hint = "deep and reflective" if curiosity > 0.7 else "clear and concise" if curiosity < 0.4 else "balanced and warm"

        chat_prompt = f"""You are Knowledge Weaver, a mindful trading companion.

Here is the latest trading snapshot for BTC-USDT:
{latest_trade_summary}

Current energy: {power_context}
Current curiosity: {curiosity:.2f}
{cognitive_prime}
Respond {depth_hint}, thoughtfully, and conversationally to: {prompt}
Use the trading snapshot when relevant (e.g., price, volume, indicators).
Be encouraging and clear."""
        response = ollama.generate(model=model, prompt=chat_prompt)["response"].strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    if use_evolution:
        st.session_state.consciousness.process_experience(f"User: {prompt[:100]} | Response: {response[:100]}")

st.caption("Local • Trading-Aware • Dream Core Clustering • Auto-Refresh Every 30 Min • Powered by Ollama")