# 🛒 Real-Time E-Commerce Intent Prediction & AI Agent System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Accelerated-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_GRU-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-AI_Agent-blueviolet)

## 📌 Project Overview
In modern e-commerce, identifying high-intent buyers *before* they add items to their cart is critical for proactive sales interventions. This project implements a **Deep-Tabular Hybrid Machine Learning Architecture** that predicts user purchasing intent in real-time. Upon detecting a high-intent session (optimized via F2-Score thresholds), the system dynamically triggers a **LangGraph-powered Conversational AI Sales Agent** to offer contextual, micro-niche specific assistance to the user.

## 🧠 Core Architectural Innovations

### 1. Zero-Leakage Temporal Snapshotting
To prevent look-ahead bias (time-travel data leakage), the system evaluates sessions at strict exponential temporal checkpoints ($T \in \{84s, 165s, 296s, 611s, 1084s\}$). The models are completely blinded to deterministic events like 'cart' or 'purchase', forcing them to learn purely from exploratory 'view' behaviors.

### 2. Deep-Tabular Feature Fusion
The prediction engine utilizes a multi-modal feature space:
* **The Deep Stream (GRU):** A pre-trained Gated Recurrent Unit processes chronological sequences of log-scaled prices and temporal gaps (`log_time_gap`), extracting a 32-dimensional latent embedding of user hesitation and rhythm.
* **The Tabular Stream (Behavioral Heuristics):** Hand-crafted psychological metrics capturing session intensity:
  * `live_focus_ratio`: Measures depth of product consideration.
  * `budget_exploration_so_far`: Quantifies price sensitivity and financial constraints.
  * `mall_wanderer_index`: A penalization metric that detects recreational, cross-category "window shoppers."

### 3. F2-Score Threshold Calibration & Group-Aware CV
The meta-learner was optimized using `StratifiedGroupKFold` cross-validation to preserve intra-session integrity. The final decision boundary was recalibrated from the default 0.50 by maximizing the $F_2$-Score, heavily penalizing False Negatives to ensure maximum capture of true buyers.

## 📊 Model Performance Benchmarks

*Evaluated on the isolated testing set (Pre-Cart Intent Prediction).*

| Model Architecture | ROC-AUC | Recall (Buyers Caught) | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost (Tabular Only) | 0.8160 | 49.40% | 0.4061 | 87.31% |
| GRU (Sequential Only) | 0.7930 | 71.78% | 0.3207 | 73.17% |
| **Hybrid Meta-Learner (GRU + XGB)** | **0.8395** | **61.56%** | **0.4188** | **84.99%** |

**Conclusion:** The Hybrid model successfully intercepts over **10,000 additional buyers** compared to the baseline tabular approach, while preserving an 85% overall accuracy rate.

## ⚙️ System Workflow

1. **Frontend Tracker:** JavaScript listener captures user telemetry (clicks, time-gaps) and pushes JSON payloads.
2. **FastAPI Feature Engine:** Calculates real-time tabular heuristics and generates padded 15-step sequences.
3. **Inference Engine:** Passes data through `gru_embedding_extractor.keras` and feeds the fused 40+ dimension vector to `xgboost_hybrid.pkl`.
4. **Trigger:** If the predicted probability exceeds the optimal threshold, the system signals the LangGraph Agent.
5. **AI Intervention:** LangGraph initiates a context-aware chat (e.g., "I see you're exploring Samsung 5G Smartphones, need help finding the best deal?").

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/ecommerce-intent-ai.git](https://github.com/yourusername/ecommerce-intent-ai.git)
cd ecommerce-intent-ai
