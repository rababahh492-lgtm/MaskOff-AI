# 🎭 MaskOff AI

 *"Trust What You See Again"*

MaskOff AI is a forensic system that detects deepfake videos by analyzing visual and temporal manipulation patterns.


 What It Does

MaskOff AI analyzes videos through multiple forensic lenses to detect signs of AI-generated manipulation. It provides a clear, explainable verdict and generates professional reports suitable for investigative use.



  Key Features

| Feature | Description |
|---------|-------------|
| **Face Analysis** | 7 detection factors: blur, symmetry, edges, colors, noise, eye region, and temporal consistency |
| **Audio Analysis** | Voice deepfake detection using MFCC and spectral analysis |
| **Evidence Gallery** | Automatically saves and displays suspicious frames |
| **Forensic Reports** | Generates detailed PDF reports with risk graphs and detection factors |
| **Interactive Dashboard** | Real-time visualization of deepfake probability over time |



# How It Works

1. **Upload** a video file (MP4, AVI, or MOV)
2. **AI Engine** analyzes faces and audio track
3. **Results** show authenticity score and risk level
4. **Evidence** is displayed in an interactive gallery
5. **Report** can be downloaded as a professional PDF



# Technologies Used

- **Python 3.10**
- **Streamlit** – Interactive dashboard
- **OpenCV** – Face detection and image processing
- **Librosa** – Audio feature extraction (MFCC)
- **Plotly** – Interactive graphs
- **FPDF** – PDF report generation
- **Scikit-learn** – Analysis algorithms

---

#Installation

 1. Clone the repository

```bash
git clone https://github.com/rababahh492-lgtm/MaskOff-AI.git
cd MaskOff-AI

2. Create and activate virtual environment:
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Run the application:
streamlit run app.py

5. Open your browser
Navigate to: http://localhost:8501


📁 Project Structure

MaskOff-AI/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── models/                # Detection models
├── utils/                 # Helper modules
├── training/              # Training scripts
├── evidence/              # Saved suspicious frames
├── reports/               # Generated PDF reports
└── uploads/               # Uploaded videos

Sample Report
The forensic PDF report includes:

Executive summary with verdict

Risk assessment table

Deepfake probability timeline graph

Detection factors and reasons

List of top suspicious frames


