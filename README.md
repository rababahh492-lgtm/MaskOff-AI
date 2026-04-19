# 🎭 MaskOff AI

## *"Trust What You See Again"*

### AI Forensic Platform | Deepfake Investigation Tool

---

## 📖 **What is MaskOff AI?**

**MaskOff AI is a forensic system that detects deepfake videos.**

It analyzes videos through **multiple lenses**:
- 👤 **Face Analysis** (7 visual factors)
- 🎤 **Audio Analysis** (voice patterns)
- ⏱️ **Temporal Analysis** (motion consistency)

Then it gives you:
- 📊 A clear **risk score** (0-100%)
- 🔍 **Why** it made that decision
- 🖼️ **Evidence gallery** of suspicious frames
- 📄 **Professional PDF report**

---

## 🎯 **Why I built this**

Deepfakes are everywhere. Anyone can make a video of you saying something you never said.

I wanted to build a tool that helps:
- 🗞️ Journalists verify video evidence
- ⚖️ Investigators analyze digital evidence
- 🔒 Security teams protect their executives
- 👀 Anyone who wants to know if a video is real

---

## ⚙️ **How It Works**

### Step by step:

1. **You upload a video** (or use live camera)
2. **The system extracts frames** and detects faces
3. **7 forensic tests** run on each face:
   - Blur detection
   - Facial symmetry
   - Edge artifacts
   - Color analysis
   - Noise patterns
   - Eye region
   - Motion consistency
4. **Audio is analyzed** (MFCC, spectral features)
5. **All results are combined** with dynamic weights
6. **You get a verdict** + reasons + evidence + PDF report

---

## 🛠️ **Technologies I used**

| Technology | What for |
|------------|----------|
| **Python 3.10** | The main language |
| **Streamlit** | Interactive dashboard |
| **OpenCV** | Face detection and image processing |
| **Librosa** | Audio analysis (MFCC) |
| **Plotly** | Interactive graphs |
| **FPDF** | PDF report generation |
| **TensorFlow** | AI model (future training) |

---

## 🚀 **How to run it**

### 1. Clone the repository

```bash
git clone https://github.com/rababahh492-lgtm/MaskOff-AI.git
cd MaskOff-AI

2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py

5. Open your browser
Go to: http://localhost:8501

📁 Project Structure

MaskOff-AI/
├── app.py                 # Main application
├── core/                  # Detection & risk logic
│   ├── detector.py       # 7-factor face analysis
│   ├── risk_engine.py    # Dynamic risk scoring
│   └── explainability.py # Top reasons extraction
├── utils/                 # Helpers
│   ├── audio_analysis.py # MFCC audio analysis
│   ├── evidence_manager.py
│   └── report_generator.py
├── evidence/              # Saved suspicious frames
├── reports/               # PDF reports
└── uploads/               # Uploaded videos

📊 What the output looks like
When you analyze a video, you get:

5 metric cards: FRAMES, SUSPICIOUS, AUTHENTICITY, UNIFIED RISK, MODEL

Top 3 reasons explaining the verdict

Interactive graph showing risk over time

Evidence gallery with suspicious frames

PDF report you can download

🎥 What you can do
Feature	Description
Upload video	Analyze any MP4, AVI, or MOV file
Live camera	Real-time deepfake detection
Real-time analysis	See frame-by-frame results
Evidence gallery	Save and view suspicious frames
PDF report	Generate forensic reports
🔬 The 7 face analysis factors
Factor	What it detects
Blur	Unnatural smoothness (common in deepfakes)
Symmetry	Facial asymmetry (real faces are symmetric)
Edges	Blending artifacts (where fake meets real)
Colors	Abnormal skin tones or saturation
Noise	Inconsistent noise patterns
Eyes	Missing reflections or unnatural details
Temporal	Sudden changes between frames
🎤 Audio analysis
The system extracts audio from the video and analyzes:

MFCC features (voice fingerprint)

Spectral centroid (frequency distribution)

Zero crossing rate (speech naturalness)

RMS energy (voice consistency)

If the audio shows anomalies, it contributes to the final risk score.

🧠 Dynamic risk scoring
Not all factors are equally important. The system uses dynamic weights:

If audio quality is poor → reduce audio weight, increase face weight

If face is blurry → reduce face weight, distribute to others

If temporal data is unstable → reduce temporal weight

This makes the system adaptive and more accurate.

📄 Forensic PDF report
The PDF includes:

Executive summary with verdict

Unified risk score (0-100%)

Deepfake probability timeline graph

Top detection factors

List of suspicious frames with details

🚧 What's next?
Face analysis (7 factors)

Audio analysis (MFCC)

Dynamic risk scoring

Explainable AI

Evidence gallery

PDF reports

Live camera detection

Deep learning model training

Cloud API deployment

💬 A personal note
This project started as a graduation project, but it became something I'm truly proud of. I built it because I believe that in an era where seeing is no longer believing, we need tools to restore trust.

MaskOff AI is my small contribution to that mission.

Thank you for reading. 🎭
