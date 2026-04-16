🎭 MaskOff AI
Trust What You See Again

A practical deepfake detection system designed as a real-world forensic tool — not just a machine learning model.


## Overview:

MaskOff AI is a hybrid forensic system that detects deepfake videos by analyzing visual, temporal, and audio manipulation patterns.

It combines multiple detection layers to provide a clear, explainable verdict, along with forensic evidence and professional reports suitable for real-world investigation scenarios.


## Why It Matters:

In an era where synthetic media is becoming indistinguishable from reality, MaskOff AI aims to restore trust in digital content by providing transparent and explainable forensic analysis.

### Key Features
## Multi-Factor Face Analysis

Advanced visual forensics including:

Blur detection
Facial symmetry analysis
Edge artifacts detection
Color inconsistencies
Noise patterns
Eye-region anomalies
Temporal consistency across frames


## Audio Analysis (Experimental)

MFCC-based feature extraction and spectral analysis for detecting potential voice manipulation.


🧾 Explainability Layer

Provides human-readable reasons behind each detection decision (Top suspicious indicators).


🖼️ Evidence Gallery

Automatically captures and displays suspicious frames for visual inspection.


📊 Interactive Dashboard

Real-time visualization of deepfake probability across video frames.


📹 Live Camera Detection

Real-time deepfake detection using webcam feed with instant AI-based analysis.


🎯 Unified Risk Score

A combined forensic score (0–100%) that evaluates visual, temporal, and audio manipulation signals across multiple detection layers.


📄 Forensic Report Generator

Generates professional PDF reports including:

Executive summary
Risk assessment
Timeline graphs
Detection reasoning
Suspicious frame evidence


🔬 How It Works
Upload a video file (MP4, AVI, MOV)
Hybrid analysis engine processes:
Face features
Temporal patterns
Audio signals
System calculates:
Authenticity Score
Unified Risk Score
Suspicious frames are extracted and stored
Results are visualized in an interactive dashboard
A full forensic report can be generated and downloaded


🧰 Technologies Used
Python 3.10
Streamlit – Interactive UI
OpenCV – Face detection & image processing
Librosa – Audio feature extraction (MFCC)
Plotly – Data visualization
FPDF – PDF report generation
Scikit-learn – Analytical modeling


🚀 Installation
# 1. Clone repository
git clone https://github.com/rababahh492-lgtm/MaskOff-AI.git
cd MaskOff-AI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

Then open:

http://localhost:8501


📁 Project Structure
MaskOff-AI/
├── app.py                 # Main application
├── core/                 # Core detection & risk logic
├── models/               # Detection models
├── utils/                # Helper modules
├── training/             # Training scripts
├── evidence/             # Saved suspicious frames
├── reports/              # Generated reports
├── uploads/              # Uploaded videos
├── weights/              # Model weights
├── requirements.txt      # Dependencies
└── README.md


📊 Sample Report Includes
Executive summary with final verdict
Unified risk score
Deepfake probability timeline
Key detection factors
Top suspicious frames

___ Future Improvements
Integration of deep learning models (EfficientNet / CNN)
Lip-sync inconsistency detection
Real-time streaming optimization
Training on large-scale datasets (FaceForensics++)
Enhanced audio deepfake detection


#### Final Note

MaskOff AI is built to bridge the gap between academic models and real-world forensic tools, focusing on explainability, usability, and practical deployment.
