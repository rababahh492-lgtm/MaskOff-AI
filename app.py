import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image
import tempfile

# Page config
st.set_page_config(
    page_title="MaskOff AI - Deepfake Forensics",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS (نفس التصميم الأصلي)
# =========================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 25%, #141428 50%, #1a0a2e 75%, #0a0a15 100%);
    }
    
    /* Hero Section */
    .hero-section {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2rem;
    }
    
    /* Title */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 70%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Typing effect */
    .typing-wrapper {
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid #9b59b6;
        animation: typing 3.5s steps(40, end), blinkCursor 0.75s step-end infinite;
    }
    
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes blinkCursor {
        50% { border-color: transparent; }
    }
    
    .tagline-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #c4d0ff, #e0c3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Analysis Container */
    .analysis-container {
        background: rgba(10,10,26,0.8);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid rgba(102,126,234,0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102,126,234,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102,126,234,0.5);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Risk badges */
    .risk-high {
        background: #ff4757;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-medium {
        background: #ffa502;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-low {
        background: #2ed573;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'video_name' not in st.session_state:
    st.session_state.video_name = None

# =========================
# Import Modules
# =========================
from models.deepfake_detector import DeepfakeDetector
from utils.evidence_manager import EvidenceSystem
from utils.report_generator import ForensicReportGenerator
from utils.audio_analysis import AudioAnalyzer

# Initialize components
detector = DeepfakeDetector()
evidence = EvidenceSystem()
report_gen = ForensicReportGenerator()
audio_analyzer = AudioAnalyzer()

# =========================
# Helper Functions
# =========================

def calculate_unified_risk(face_score, audio_score, temporal_score, lip_sync_score=0.5):
    """Calculate Unified Risk Score (0-100)"""
    weights = {
        'face': 0.40,
        'audio': 0.25,
        'temporal': 0.20,
        'lip_sync': 0.15
    }
    
    total = (
        face_score * weights['face'] +
        audio_score * weights['audio'] +
        temporal_score * weights['temporal'] +
        lip_sync_score * weights['lip_sync']
    )
    
    return total * 100

def get_top_reasons(results):
    """Extract top 3 reasons for the verdict"""
    all_reasons = results.get('reasons', []).copy()
    
    if results.get('suspicious_count', 0) > results.get('analyzed_frames', 1) * 0.5:
        all_reasons.append("High number of suspicious frames")
    if results.get('authenticity_score', 100) < 50:
        all_reasons.append("Low authenticity score indicates manipulation")
    
    audio_reasons = results.get('audio_reasons', [])
    all_reasons.extend(audio_reasons)
    
    unique_reasons = []
    for r in all_reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)
    
    return unique_reasons[:3] if unique_reasons else ["Analysis complete", "No major anomalies detected"]

# =========================
# Landing Page
# =========================
if st.session_state.page == 'landing':
    st.markdown("""
    <div class="hero-section">
        <div>
            <h1 class="main-title">MASKOFF AI</h1>
        </div>
        <div style="margin: 2rem 0;">
            <div class="typing-wrapper">
                <span class="tagline-text">"Trust What You See Again"</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.2, 1.6, 1.2])
    with col2:
        if st.button("🚀 START ANALYSIS", key="start_btn"):
            st.session_state.page = 'analysis'
            st.rerun()

# =========================
# Analysis Page
# =========================
elif st.session_state.page == 'analysis':
    
    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    
    # Header with Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.page = 'landing'
            st.session_state.analysis_complete = False
            st.session_state.results = None
            st.rerun()
    
    with col2:
        st.markdown("""
        <h2 style="text-align: center; background: linear-gradient(135deg, #667eea, #764ba2);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-family: 'Orbitron', monospace;">
            🎭 MaskOff AI
        </h2>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================
    # TABS: Upload Video / Live Camera / Real-Time Video
    # =========================
    tab1, tab2, tab3 = st.tabs(["📁 Upload Video", "🎥 Live Camera", "📹 Real-Time Video Analysis"])
    
    # ========== TAB 1: Upload Video (نفس النظام القديم) ==========
    with tab1:
        if not st.session_state.analysis_complete and not st.session_state.processing:
            uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])
            
            if uploaded_file:
                os.makedirs("uploads", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{timestamp}_{uploaded_file.name}"
                video_path = os.path.join("uploads", video_filename)
                
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                st.session_state.video_path = video_path
                st.session_state.video_name = uploaded_file.name
                
                st.video(video_path)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 START ANALYSIS"):
                        st.session_state.processing = True
                        st.rerun()
    
    # ========== TAB 2: Live Camera ==========
    with tab2:
        st.markdown("### 🎥 Live Camera Deepfake Detection")
        st.markdown("Point your camera at a face and watch real-time analysis")
        
        start_camera = st.button("📸 Start Live Camera")
        
        if start_camera:
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Please check your camera connection.")
            else:
                stop = st.button("🛑 Stop Camera")
                while not stop:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    result = detector.analyze_frame(frame)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    with metrics_placeholder.container():
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Fake Probability", f"{result['fake_probability']:.1%}")
                        with col_b:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        with col_c:
                            risk = "HIGH" if result['fake_probability'] > 0.7 else "MEDIUM" if result['fake_probability'] > 0.4 else "LOW"
                            st.metric("Risk Level", risk)
                    
                    if stop:
                        break
                
                cap.release()
    
    # ========== TAB 3: Real-Time Video Analysis ==========
    with tab3:
        st.markdown("### 📹 Real-Time Video Frame Analysis")
        
        uploaded_live_video = st.file_uploader("Upload Video for Real-Time Analysis", type=["mp4", "avi", "mov"], key="live_video")
        
        if uploaded_live_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(uploaded_live_video.read())
                temp_path = tmpfile.name
            
            st.video(temp_path)
            
            if st.button("🔍 Analyze Frame by Frame"):
                cap = cv2.VideoCapture(temp_path)
                progress_bar = st.progress(0)
                results_container = st.container()
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_analyses = []
                
                for i in range(0, frame_count, 10):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    result = detector.analyze_frame(frame)
                    frame_analyses.append({
                        'frame': i,
                        'fake_prob': result['fake_probability'],
                        'confidence': result['confidence']
                    })
                    
                    with results_container.container():
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Current Frame", f"#{i}")
                        with col_b:
                            st.metric("Fake Probability", f"{result['fake_probability']:.1%}")
                    
                    progress_bar.progress(i / frame_count)
                
                cap.release()
                os.unlink(temp_path)
                
                if frame_analyses:
                    avg_fake = np.mean([f['fake_prob'] for f in frame_analyses])
                    st.success(f"✅ Analysis Complete! Average Fake Probability: {avg_fake:.1%}")
    
    # =========================
    # Processing (نفس النظام القديم)
    # =========================
    if st.session_state.processing:
        with st.spinner("🧠 Analyzing video with AI..."):
            cap = cv2.VideoCapture(st.session_state.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frame_analyses = []
            suspicious_frames_data = []
            frame_num = 0
            frame_skip = max(1, frame_count // 50)
            
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % frame_skip == 0:
                    result = detector.analyze_frame(frame)
                    frame_analyses.append({
                        'frame_num': frame_num,
                        'fake_prob': result['fake_probability'],
                        'confidence': result['confidence'],
                        'timestamp': frame_num / fps if fps > 0 else 0,
                        'reasons': result.get('reasons', [])
                    })
                    
                    if result['fake_probability'] > 0.5:
                        evidence_path = evidence.save_suspicious_frame(
                            frame, frame_num, result['fake_probability'], 
                            datetime.now().strftime("%H%M%S")
                        )
                        if evidence_path:
                            suspicious_frames_data.append({
                                'frame_number': frame_num,
                                'score': result['fake_probability'],
                                'confidence': result['confidence'],
                                'image_path': evidence_path,
                                'timestamp': frame_num / fps if fps > 0 else 0
                            })
                    
                    progress_bar.progress(min(1.0, frame_num / frame_count))
                
                frame_num += 1
            
            cap.release()
            
            # Audio analysis
            audio_result = audio_analyzer.detect_deepfake_audio(st.session_state.video_path)
            
            # Calculate results
            fake_probs = [a['fake_prob'] for a in frame_analyses]
            avg_fake_prob = np.mean(fake_probs) if fake_probs else 0.5
            authenticity_score = (1 - avg_fake_prob) * 100
            suspicious_count = len([p for p in fake_probs if p > 0.5])
            temporal_variance = np.var(fake_probs) if len(fake_probs) > 1 else 0.5
            
            # Unified Risk
            unified_risk = calculate_unified_risk(
                face_score=avg_fake_prob,
                audio_score=audio_result.get('fake_probability', 0.5),
                temporal_score=temporal_variance
            )
            
            # Collect reasons
            all_reasons = []
            if suspicious_count > len(frame_analyses) * 0.5:
                all_reasons.append(f"High number of suspicious frames ({suspicious_count}/{len(frame_analyses)})")
            if authenticity_score < 50:
                all_reasons.append("Low authenticity score indicates possible manipulation")
            
            for analysis in frame_analyses[:10]:
                if analysis.get('reasons'):
                    all_reasons.extend(analysis['reasons'])
            
            if audio_result.get('reasons'):
                all_reasons.extend(audio_result['reasons'])
            
            all_reasons = list(dict.fromkeys(all_reasons))[:10]
            
            st.session_state.results = {
                'frame_count': frame_count,
                'analyzed_frames': len(frame_analyses),
                'suspicious_count': suspicious_count,
                'authenticity_score': authenticity_score,
                'unified_risk': unified_risk,
                'frame_analyses': frame_analyses,
                'suspicious_frames_data': suspicious_frames_data,
                'video_name': st.session_state.video_name,
                'model_used': 'Enhanced Face + Audio Analysis',
                'reasons': all_reasons,
                'audio_analysis': audio_result,
                'audio_reasons': audio_result.get('reasons', [])
            }
        
        st.session_state.processing = False
        st.session_state.analysis_complete = True
        st.rerun()
    
    # =========================
    # Show Results
    # =========================
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        
        # Metrics - 5 cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div>🔍 FRAMES</div>
                <div class="metric-value">{results['analyzed_frames']}</div>
                <div>of {results['frame_count']} total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            suspicious = results['suspicious_count']
            risk_class = "risk-high" if suspicious > 30 else "risk-medium" if suspicious > 10 else "risk-low"
            st.markdown(f"""
            <div class="metric-card">
                <div>⚠️ SUSPICIOUS</div>
                <div class="metric-value">{suspicious}</div>
                <div><span class="{risk_class}">{suspicious/results['analyzed_frames']*100:.0f}% of frames</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            score = results['authenticity_score']
            risk_class = "risk-low" if score > 70 else "risk-medium" if score > 40 else "risk-high"
            st.markdown(f"""
            <div class="metric-card">
                <div>✅ AUTHENTICITY</div>
                <div class="metric-value">{score:.1f}%</div>
                <div><span class="{risk_class}">Trust Score</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            unified = results.get('unified_risk', 50)
            unified_class = "risk-high" if unified > 60 else "risk-medium" if unified > 30 else "risk-low"
            st.markdown(f"""
            <div class="metric-card">
                <div>🎯 UNIFIED RISK</div>
                <div class="metric-value">{unified:.0f}%</div>
                <div><span class="{unified_class}">Overall Score</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div>🧠 MODEL</div>
                <div class="metric-value" style="font-size: 1rem;">Face + Audio</div>
                <div>7 factors</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Top 3 Reasons
        top_reasons = get_top_reasons(results)
        st.markdown("---")
        st.markdown("### 🔍 Why This Verdict? (Top 3 Reasons)")
        for i, reason in enumerate(top_reasons[:3]):
            st.markdown(f"{i+1}. {reason}")
        
        # Chart
        if results['frame_analyses']:
            st.markdown("---")
            st.markdown("### 📈 Deepfake Probability Over Time")
            
            frame_data = pd.DataFrame(results['frame_analyses'])
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=frame_data['frame_num'],
                y=frame_data['fake_prob'],
                mode='lines+markers',
                name='Deepfake Probability',
                line=dict(color='#ff4757', width=2),
                marker=dict(size=6)
            ))
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="#ffa502", annotation_text="Threshold (50%)")
            
            fig.update_layout(
                title="Frame-by-Frame Deepfake Detection",
                xaxis_title="Frame Number",
                yaxis_title="Fake Probability",
                yaxis_range=[0, 1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Gallery
        if results['suspicious_frames_data']:
            st.markdown("---")
            st.markdown("### 🖼️ Suspicious Frames Gallery")
            st.markdown(f"*Found {len(results['suspicious_frames_data'])} suspicious frames*")
            
            cols = st.columns(3)
            for idx, evidence in enumerate(results['suspicious_frames_data'][:6]):
                with cols[idx % 3]:
                    if os.path.exists(evidence['image_path']):
                        try:
                            image = Image.open(evidence['image_path'])
                            st.image(image, use_container_width=True)
                            st.caption(f"Frame #{evidence['frame_number']} | Risk: {evidence['score']:.1%}")
                        except:
                            pass
        
        # PDF Report
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📑 GENERATE PDF REPORT", use_container_width=True):
                with st.spinner("Generating forensic report..."):
                    report_path = report_gen.generate_forensic_report(results, results['video_name'])
                    if report_path and os.path.exists(report_path):
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="⬇️ DOWNLOAD REPORT",
                                data=f,
                                file_name=os.path.basename(report_path),
                                mime="application/pdf",
                                use_container_width=True
                            )
        
        # Action Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.results = None
                st.rerun()
        with col2:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 'landing'
                st.session_state.analysis_complete = False
                st.session_state.results = None
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)