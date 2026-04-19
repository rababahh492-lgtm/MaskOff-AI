import cv2
import numpy as np
import os
import tensorflow as tf

class DeepfakeDetector:
    def __init__(self):
        self.face_detector = None
        self.input_size = (224, 224)
        self.face_history = []
        self.ai_model = None
        self.load_ai_model()
    
    def load_ai_model(self):
        """Load or create AI model for deepfake detection"""
        model_path = "weights/deepfake_model.h5"
        if os.path.exists(model_path):
            try:
                self.ai_model = tf.keras.models.load_model(model_path)
                print("✅ AI Model loaded from disk")
                return True
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        
        # إذا ما في نموذج، نستخدم Heuristics فقط
        self.ai_model = None
        print("⚠️ No AI model found, using enhanced heuristics only")
        return False
    
    def extract_faces(self, frame):
        """Extract face from frame"""
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                face = cv2.resize(face, self.input_size)
                self.face_history.append(face)
                if len(self.face_history) > 30:
                    self.face_history.pop(0)
                return face
        return None
    
    def analyze_with_ai(self, face):
        """Analyze face with AI model (if available)"""
        if self.ai_model is None:
            return None
        
        try:
            face_input = np.expand_dims(face / 255.0, axis=0)
            prediction = self.ai_model.predict(face_input, verbose=0)
            fake_prob = float(prediction[0][0])
            
            return {
                'fake_probability': fake_prob,
                'real_probability': 1 - fake_prob,
                'confidence': abs(fake_prob - 0.5) * 2,
                'model_used': 'AI CNN Model',
                'reasons': ['AI-based deepfake detection']
            }
        except:
            return None
    
    def analyze_face_quality(self, face):
        """Enhanced heuristic analysis (7 factors)"""
        if face is None:
            return 0.5, ["No face detected"], 0.3
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Blur Detection (وضوح الصورة)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:
            blur_score = 0.35
            blur_reason = "Very blurry (suspicious)"
        elif laplacian_var < 150:
            blur_score = 0.15
            blur_reason = "Blurry"
        else:
            blur_score = 0.05
            blur_reason = "Sharp"
        
        # 2. Symmetry Analysis (تماثل الوجه)
        if w >= 2:
            left = gray[:, :w//2]
            right = cv2.flip(gray[:, w//2:], 1)
            if left.shape == right.shape:
                symmetry_score = 1 - (np.mean(np.abs(left.astype(float) - right.astype(float))) / 255)
            else:
                symmetry_score = 0.5
        else:
            symmetry_score = 0.5
        
        if symmetry_score < 0.6:
            sym_score = 0.35
            sym_reason = "Very asymmetric (deepfake sign)"
        elif symmetry_score < 0.7:
            sym_score = 0.20
            sym_reason = "Asymmetric"
        elif symmetry_score < 0.8:
            sym_score = 0.10
            sym_reason = "Slightly asymmetric"
        else:
            sym_score = 0.05
            sym_reason = "Symmetric"
        
        # 3. Edge Artifacts (حواف غير طبيعية)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        if edge_density > 0.15:
            edge_score = 0.30
            edge_reason = "Too many edges (artifact)"
        elif edge_density > 0.10:
            edge_score = 0.15
            edge_reason = "High edge density"
        elif edge_density < 0.02:
            edge_score = 0.20
            edge_reason = "Too few edges (blending)"
        else:
            edge_score = 0.05
            edge_reason = "Normal edges"
        
        # 4. Color Analysis (تحليل الألوان)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_hue = np.mean(hsv[:, :, 0])
        skin_sat = np.mean(hsv[:, :, 1])
        
        color_score = 0
        color_reasons = []
        if skin_hue < 5 or skin_hue > 20:
            color_score += 0.15
            color_reasons.append("Abnormal skin hue")
        if skin_sat < 50 or skin_sat > 150:
            color_score += 0.10
            color_reasons.append("Abnormal saturation")
        color_score = min(color_score, 0.25)
        
        # 5. Noise Analysis (تحليل الضوضاء)
        noise = np.std(gray) / 255
        if noise < 0.03:
            noise_score = 0.20
            noise_reason = "Too smooth (fake)"
        elif noise > 0.20:
            noise_score = 0.15
            noise_reason = "Too noisy"
        else:
            noise_score = 0.05
            noise_reason = "Normal noise"
        
        # 6. Eye Region Analysis (تحليل العينين)
        eye_region = gray[h//3:h//2, :]
        if eye_region.size > 0:
            eye_variance = np.var(eye_region)
            eye_edges = cv2.Canny(eye_region, 50, 150)
            eye_edge_density = np.sum(eye_edges > 0) / eye_edges.size if eye_edges.size > 0 else 0
            
            eye_score = 0
            eye_reasons = []
            if eye_variance < 300:
                eye_score += 0.10
                eye_reasons.append("Eyes lack detail")
            if eye_edge_density > 0.20 or eye_edge_density < 0.02:
                eye_score += 0.10
                eye_reasons.append("Abnormal eye edges")
            eye_score = min(eye_score, 0.20)
        else:
            eye_score = 0.10
            eye_reasons = ["No eye region detected"]
        
        # 7. Temporal Consistency (اتساق الحركة بين الفريمات)
        if len(self.face_history) >= 5:
            prev_faces = list(self.face_history)[-5:]
            prev_avg = np.mean(prev_faces, axis=0)
            diff = np.mean(np.abs(face.astype(float) - prev_avg)) / 255
            if diff > 0.20:
                temporal_score = 0.25
                temporal_reason = "Sudden face change (temporal inconsistency)"
            elif diff > 0.10:
                temporal_score = 0.10
                temporal_reason = "Minor face changes"
            else:
                temporal_score = 0.02
                temporal_reason = "Consistent"
        else:
            temporal_score = 0.05
            temporal_reason = "Insufficient history"
        
        # حساب النتيجة النهائية (مو عشوائية)
        total_score = (
            blur_score * 0.20 +
            sym_score * 0.20 +
            edge_score * 0.15 +
            color_score * 0.15 +
            noise_score * 0.10 +
            eye_score * 0.10 +
            temporal_score * 0.10
        )
        
        # تجميع الأسباب
        all_reasons = []
        if blur_score > 0.10:
            all_reasons.append(blur_reason)
        if sym_score > 0.10:
            all_reasons.append(sym_reason)
        if edge_score > 0.10:
            all_reasons.append(edge_reason)
        all_reasons.extend(color_reasons)
        if noise_score > 0.10:
            all_reasons.append(noise_reason)
        all_reasons.extend(eye_reasons)
        all_reasons.append(temporal_reason)
        
        # نضبط النتيجة تكون بين 0 و 1
        fake_prob = max(0.1, min(0.95, total_score))
        confidence = abs(fake_prob - 0.5) * 2
        
        return fake_prob, all_reasons[:5], confidence
    
    def analyze_frame(self, frame):
        """Analyze single frame - AI first, then heuristics"""
        face = self.extract_faces(frame)
        
        if face is not None:
            # Try AI first if available
            ai_result = self.analyze_with_ai(face)
            if ai_result:
                return ai_result
        
        # Fallback to enhanced heuristics
        if face is not None:
            fake_prob, reasons, confidence = self.analyze_face_quality(face)
            return {
                'fake_probability': float(fake_prob),
                'real_probability': float(1 - fake_prob),
                'confidence': float(confidence),
                'model_used': 'Enhanced Heuristics (7 factors)',
                'reasons': reasons
            }
        else:
            return {
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'confidence': 0.3,
                'model_used': 'No Face Detected',
                'reasons': ['No face detected in frame', 'Try a video with clear faces']
            }
    
    def analyze_sequence(self, frames_sequence, fps=30):
        """Analyze sequence of frames"""
        if len(frames_sequence) < 5:
            return None
        
        fake_scores = []
        all_reasons = []
        
        for frame in frames_sequence[:min(30, len(frames_sequence))]:
            result = self.analyze_frame(frame)
            fake_scores.append(result['fake_probability'])
            all_reasons.extend(result.get('reasons', []))
        
        avg_score = np.mean(fake_scores)
        variance = np.var(fake_scores)
        
        if variance > 0.08:
            avg_score = min(avg_score + 0.12, 0.95)
            all_reasons.append("High temporal variance detected")
        
        unique_reasons = list(dict.fromkeys(all_reasons))[:5]
        
        return {
            'fake_probability': float(avg_score),
            'real_probability': float(1 - avg_score),
            'confidence': float(min(abs(avg_score - 0.5) * 2, 0.95)),
            'model_used': 'Temporal Analysis',
            'variance': float(variance),
            'reasons': unique_reasons
        }
    
    def build_spatial_model(self):
        return None
    
    def build_temporal_model(self):
        return None
