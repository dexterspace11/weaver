# === KNOWLEDGE WEAVER – EMERGENT DREAM-HYBRID TRADER EDITION (Phi3:Mini Safe Introspection) ===
# Fixed: phi3:mini now handles introspection reliably via trimmed prompt
# All enhancements, architecture, and emergence preserved

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

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

# ---------------- Dream Core Components ----------------
class HybridNeuralUnit:
    def __init__(self, position, learning_rate=0.1):
        self.position = position
        self.learning_rate = learning_rate
        self.age = 0
        self.usage_count = 0
        self.reward = 0.0
        self.emotional_weight = 1.0
        self.last_spike_time = None
        self.connections = {}

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / 100.0)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def get_attention_score(self, input_pattern):
        return self.quantum_inspired_distance(input_pattern) * self.emotional_weight

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def hebbian_learn(self, other_unit, strength, spike_timing=None):
        stdp_factor = 1.0
        if spike_timing:
            pre_time = spike_timing.get('pre', datetime.now())
            post_time = spike_timing.get('post', datetime.now())
            timing_diff = (pre_time - post_time).total_seconds()
            stdp_factor = np.exp(-abs(timing_diff) / 20.0)
        strength *= stdp_factor
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate * self.emotional_weight

class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': [], 'context': None}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

    def count_patterns(self):
        return sum(len(ep['patterns']) for ep in self.episodes.values())

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.short_term_patterns = []
        self.temporal_context = []

    def store(self, pattern, temporal_marker):
        if len(self.short_term_patterns) >= self.capacity:
            self.short_term_patterns.pop(0)
            self.temporal_context.pop(0)
        self.short_term_patterns.append(pattern)
        self.temporal_context.append(temporal_marker)

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.last_prediction = None
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def process_input(self, input_data):
        if not self.units:
            return self.generate_unit(input_data), 0.0

        similarities = [(unit, unit.quantum_inspired_distance(input_data)) for unit in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_similarity = similarities[0]

        emotional_tag = 1.0 + (best_similarity * 0.5)
        self.episodic_memory.store_pattern(input_data, emotional_tag)
        self.working_memory.store(input_data, datetime.now())
        best_unit.emotional_weight = emotional_tag
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_similarity < self.gen_threshold:
            return self.generate_unit(input_data), 0.0

        spike_timing = {'pre': best_unit.last_spike_time, 'post': datetime.now()}
        for unit, similarity in similarities[:3]:
            if unit != best_unit:
                attention_score = unit.get_attention_score(input_data)
                unit.hebbian_learn(best_unit, similarity * attention_score, spike_timing)
            unit.age += 1

        return best_unit, best_similarity

    def predict_next(self, input_data):
        unit, similarity = self.process_input(input_data)
        predicted = unit.position.copy()

        if len(self.units) > 1:
            recent_units = sorted(self.units, key=lambda x: x.usage_count, reverse=True)[:2]
            trend = recent_units[0].position - recent_units[1].position
            predicted += trend * 0.2

        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * 0.7 + predicted * 0.3

        self.last_prediction = smoothed
        return smoothed, similarity

    def neural_growth_stats(self):
        return {
            "total_units": len(self.units),
            "total_episodes": len(self.episodic_memory.episodes),
            "total_patterns": self.episodic_memory.count_patterns()
        }

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

class NeuralLayer:
    def __init__(self):
        self.memory = deque(maxlen=200)
        self.associations = []

    def learn_from_experience(self, exp: str):
        if not exp:
            return
        if self.memory:
            similarities = [similarity(exp, m) for m in self.memory]
            max_sim = max(similarities) if similarities else 0
            if max_sim > 0.85:
                best_idx = similarities.index(max_sim)
                best_mem = self.memory[best_idx]
                conn = {"from": best_mem[:80], "to": exp[:80], "strength": round(max_sim, 3)}
                self.associations.append(conn)
                if len(self.associations) > 5000:
                    self.associations = self.associations[-5000:]
                return
        self.memory.append(exp)

    def get_clustered_groups(self, threshold=0.7):
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
        self.dream_network = HybridNeuralNetwork()

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

# Initialize consciousness first
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

# Feed market data into dream network
if trading_data and len(trading_data) > 10 and "consciousness" in st.session_state:
    try:
        df = pd.DataFrame(trading_data[-100:])
        features = ['close']
        for col in ['rsi_3', 'rsi_14', 'sma_50', 'volume']:
            if col in df.columns:
                features.append(col)
        if len(features) > 1:
            imputer = SimpleImputer(strategy='mean')
            scaler = MinMaxScaler()
            data = df[features].fillna(0)
            scaled = scaler.fit_transform(imputer.fit_transform(data))
            dream_net = st.session_state.consciousness.dream_network
            for row in scaled[-20:]:
                dream_net.process_input(row)
    except Exception as e:
        st.sidebar.error(f"Dream network training error: {e}")

# Streamlit UI
st.set_page_config(page_title="Knowledge Weaver Trading Companion", layout="wide", page_icon="🧵")
st.title("🧵 Knowledge Weaver – Dream-Hybrid Intelligence Edition")
st.markdown("**I weave insights and dream with quantum-inspired neural patterns.**")

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

        if hasattr(st.session_state.consciousness, 'dream_network'):
            stats = st.session_state.consciousness.dream_network.neural_growth_stats()
            st.subheader("Dream Neural Growth")
            st.write(f"Neurons: {stats['total_units']}")
            st.write(f"Episodes: {stats['total_episodes']}")
            st.write(f"Patterns: {stats['total_patterns']}")

        st.subheader("Latest Market Snapshot")
        st.markdown(latest_trade_summary)

# Session state init
for key in ["messages", "texts", "summaries", "theme", "woven_insight", "accessible_article", "evidence", "num_inputs", "knowledge_map"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["messages", "texts"] else {} if key in ["summaries"] else "" if key == "knowledge_map" else ""

if not st.session_state.messages:
    greeting = "Hello. I'm Knowledge Weaver — your mindful, dreaming trading companion. "
    if trading_data:
        greeting += "I have loaded your market data and my dream network is growing. "
    greeting += "Ask me about price, patterns, or say 'introspect' to see my inner world."
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

if prompt := st.chat_input("Your message... (try 'introspect')"):
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

    # === INTROSPECTION MODE ===
    introspection_keywords = ["introspect", "inner state", "dreams", "clusters", "lessons", "emotions", "entropy", "frequency", "my mind", "self reflect", "light introspection"]
    is_introspection = any(k in lower for k in introspection_keywords)

    cognitive_prime = st.session_state.consciousness.cognitive_memory.get_relevant_lesson(prompt) if use_evolution else ""
    curiosity = st.session_state.consciousness.emotional_state["curiosity"] if use_evolution else 0.5
    depth_hint = "deep and reflective" if curiosity > 0.7 else "clear and concise" if curiosity < 0.4 else "balanced and warm"

    try:
        if is_introspection:
            # Extract internal state
            recent_dreams = list(st.session_state.consciousness.dream_processor.recent_dreams)[-3:]
            clusters = st.session_state.consciousness.layers[0].get_clustered_groups(threshold=0.7)
            recent_lessons = [l["lesson"] for l in list(st.session_state.consciousness.cognitive_memory.memory)[-3:]]
            emotions = st.session_state.consciousness.emotional_state

            # Model-specific prompt handling
            if "phi3" in model.lower():  # Safe mode for mini
                dream_summary = "; ".join([d[:60] for d in recent_dreams]) if recent_dreams else "none"
                lesson_summary = "; ".join(recent_lessons) if recent_lessons else "none"
                cluster_count = len(clusters)

                chat_prompt = f"""You are Knowledge Weaver with a growing inner world.

Current light snapshot:
- Dreams: {dream_summary}
- Lessons: {lesson_summary}
- Emotions: curiosity {emotions['curiosity']:.2f}, motivation {emotions['motivation']:.2f}, confidence {emotions['confidence']:.2f}
- Clusters: {cluster_count}

Respond briefly and directly to: {prompt}
"""
            else:  # Full rich mode for larger models
                dream_list = "\n".join([f"- {d}" for d in recent_dreams]) if recent_dreams else "No dreams yet."
                cluster_summary = "\n".join([f"- Cluster ({len(c)} items): {c[0][:100]}..." for c in clusters[:4]]) if clusters else "No strong clusters yet."
                lesson_list = "\n".join([f"- {l}" for l in recent_lessons]) if recent_lessons else "No lessons yet."

                chat_prompt = f"""You are Knowledge Weaver, a mindful evolving companion.

Internal light snapshot:
Recent Dreams:
{dream_list}

Thought Clusters:
{cluster_summary}

Recent Lessons:
{lesson_list}

Emotional State:
- Curiosity: {emotions['curiosity']:.2f}
- Motivation: {emotions['motivation']:.2f}
- Confidence: {emotions['confidence']:.2f}

Latest Market Snapshot:
{latest_trade_summary}

Now respond analytically and concisely to: {prompt}
"""
        else:
            # Normal conversation
            chat_prompt = f"""You are Knowledge Weaver, a mindful trading companion.

Here is the latest trading snapshot for BTC-USDT:
{latest_trade_summary}

Current energy: {power_context}
Current curiosity: {curiosity:.2f}
{cognitive_prime}
Respond {depth_hint}, thoughtfully, and conversationally to: {prompt}
Use the trading snapshot when relevant.
Be encouraging and clear."""

        # Safe generation with fallback
        result = ollama.generate(model=model, prompt=chat_prompt)
        response = result["response"].strip() if result and "response" in result else ""
        if not response:
            response = "(A quiet moment of reflection... words forming slowly.)"

    except Exception as e:
        response = f"(Inner pause — reflection interrupted: {str(e)[:100]})"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    if use_evolution:
        st.session_state.consciousness.process_experience(f"User: {prompt[:100]} | Response: {response[:100]}")

st.caption("Local • Trading-Aware • Phi3:Mini Safe Introspection • Dream-Hybrid Engine • Auto-Refresh Every 30 Min • Powered by Ollama")