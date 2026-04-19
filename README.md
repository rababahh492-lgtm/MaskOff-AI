🎭 MaskOff AI
“Trust What You See Again”

AI Forensic Platform | Deepfake Investigation Tool

📌 Overview

MaskOff AI is a multimodal forensic system designed to detect and analyze deepfake videos using AI-driven analysis across multiple signals.

It evaluates videos through three main perspectives:

👤 Face Analysis (7 forensic visual factors)
🎤 Audio Analysis (voice authenticity & consistency)
⏱️ Temporal Analysis (frame-to-frame stability)

The system then produces a unified forensic output including:

📊 Risk Score (0–100%)
🔍 Explainable reasoning (why the decision was made)
🖼️ Evidence gallery (suspicious frames)
📄 Professional PDF forensic report
🎯 Problem Statement

With the rise of deepfake technology, visual content can no longer be trusted blindly. Videos can now be manipulated to:

Fake speech
Alter facial expressions
Synthesize realistic voices

MaskOff AI aims to help restore digital trust by providing a forensic layer for video verification.

⚙️ How It Works
User uploads a video or uses live camera
Video is decomposed into frames
Each frame is analyzed using 7 facial forensic checks
Audio track is extracted and analyzed separately
Temporal consistency is measured across frames
A dynamic risk engine fuses all signals
Explainability module generates human-readable reasons
Final forensic report is produced
🧠 Core Modules
👤 Face Analysis (7 Forensic Factors)
Blur detection (GAN smoothing artifacts)
Facial symmetry consistency
Edge blending detection
Color distribution anomalies
Noise pattern inconsistencies
Eye region artifacts (reflection & realism)
Motion consistency across frames
🎤 Audio Analysis

Audio is extracted and analyzed using signal processing:

MFCC features (voice fingerprint)
Spectral centroid (frequency distribution)
Zero-crossing rate (speech naturalness)
RMS energy (voice stability)

👉 Detects synthetic or cloned voice patterns

⏱️ Temporal Analysis

Detects inconsistencies over time using:

Variance tracking
Spike detection (Z-score)
Frame probability fluctuations
🧠 Risk Engine (Adaptive Fusion)

Instead of fixed weights, MaskOff AI uses a dynamic weighting system:

Low-quality audio → reduce audio weight
Blurry face → reduce face confidence
Unstable frames → adjust temporal impact

This creates a more realistic and adaptive risk score.

🔍 Explainable AI Layer

The system does not only detect deepfakes — it explains them.

It provides:

Ranked forensic reasons
Weighted importance of each signal
Human-readable explanations

👉 Ensures transparency instead of black-box decisions

📊 Output

Each analysis generates:

🎯 Unified Risk Score (0–100%)
🔍 Top 3 forensic reasons
📈 Probability timeline graph
🖼️ Suspicious frame evidence
📄 PDF forensic report
🛠️ Tech Stack
Technology	Purpose
Python 3.10	Core system
Streamlit	Web interface
OpenCV	Video & frame processing
Librosa	Audio feature extraction
Plotly	Visualization
FPDF	PDF report generation

🚀 Installation
git clone https://github.com/rababahh492-lgtm/MaskOff-AI.git
cd MaskOff-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Then open:

http://localhost:8501

📁 Project Structure
MaskOff-AI/
│
├── app.py
├── core/
│   ├── detector.py
│   ├── risk_engine.py
│   └── explainability.py
│
├── utils/
│   ├── audio_analysis.py
│   ├── evidence_manager.py
│   └── report_generator.py
│
├── evidence/
├── reports/
└── uploads/
🎥 Features
📤 Upload video analysis
🎥 Live camera detection
📹 Frame-by-frame inspection
🖼️ Evidence gallery
📄 PDF forensic reports
📊 Interactive graphs
🔬 Future Work
Deep learning model training (CNN / Transformer)
Cloud API deployment
Real-time browser plugin
Large-scale dataset integration
Voice cloning detection improvement

💬 Personal Note

MaskOff AI started as a graduation project, but evolved into a forensic AI system focused on restoring trust in digital media.

In a world where seeing is no longer believing — tools like this become necessary.

🎭 MaskOff AI

Built to help people distinguish reality from manipulation.
