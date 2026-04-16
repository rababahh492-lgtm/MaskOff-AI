"""
Deepfake Detector Module
Main detection logic
"""

import cv2
import numpy as np
import os

class DeepfakeDetector:
    def __init__(self):
        self.face_detector = None
        self.input_size = (224, 224)
        self.face_history = []
        
    def _init_face_detector(self):
        """Lazy initialization of face detector"""
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def extract_faces(self, frame):
        """Extract face from frame"""
        self._init_face_detector()
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
    
    def analyze_blur(self, gray):
        """Blur detection"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 80:
            return 0.35, "Very blurry (suspicious)"
        elif laplacian_var < 150:
            return 0.15, "Blurry"
        else:
            return 0.05, "Sharp"
    
    def analyze_symmetry(self, gray):
        """Symmetry analysis"""
        h, w = gray.shape
        if w < 2:
            return 0.5, "Too small"
        
        left = gray[:, :w//2]
        right = cv2.flip(gray[:, w//2:], 1)
        
        if left.shape != right.shape:
            return 0.5, "Asymmetric shape"
        
        diff = np.mean(np.abs(left.astype(float) - right.astype(float))) / 255
        symmetry = 1 - diff
        
        if symmetry < 0.6:
            return 0.35, "Very asymmetric (deepfake sign)"
        elif symmetry < 0.7:
            return 0.20, "Asymmetric"
        elif symmetry < 0.8:
            return 0.10, "Slightly asymmetric"
        else:
            return 0.05, "Symmetric"
    
    def analyze_edges(self, gray):
        """Edge artifacts detection"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        if edge_density > 0.15:
            return 0.30, "Too many edges (artifact)"
        elif edge_density > 0.10:
            return 0.15, "High edge density"
        elif edge_density < 0.02:
            return 0.20, "Too few edges (blending)"
        else:
            return 0.05, "Normal"
    
    def analyze_color(self, face):
        """Color analysis"""
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        skin_hue = np.mean(hsv[:, :, 0])
        skin_sat = np.mean(hsv[:, :, 1])
        
        score = 0
        reasons = []
        
        if skin_hue < 5 or skin_hue > 20:
            score += 0.15
            reasons.append("Abnormal skin hue")
        
        if skin_sat < 50 or skin_sat > 150:
            score += 0.10
            reasons.append("Abnormal saturation")
        
        return min(score, 0.30), reasons if reasons else ["Normal colors"]
    
    def analyze_noise(self, gray):
        """Noise analysis"""
        noise = np.std(gray) / 255
        
        if noise < 0.03:
            return 0.20, "Too smooth (fake)"
        elif noise > 0.20:
            return 0.15, "Too noisy"
        else:
            return 0.05, "Normal"
    
    def analyze_eye_region(self, gray):
        """Eye region analysis"""
        h, w = gray.shape
        eye_region = gray[h//3:h//2, :]
        
        if eye_region.size == 0:
            return 0.10, "No eye region"
        
        eye_variance = np.var(eye_region)
        eye_edges = cv2.Canny(eye_region, 50, 150)
        eye_edge_density = np.sum(eye_edges > 0) / eye_edges.size if eye_edges.size > 0 else 0
        
        score = 0
        reasons = []
        
        if eye_variance < 300:
            score += 0.10
            reasons.append("Eyes lack detail")
        
        if eye_edge_density > 0.20 or eye_edge_density < 0.02:
            score += 0.10
            reasons.append("Abnormal eye edges")
        
        return min(score, 0.20), reasons if reasons else ["Normal eyes"]
    
    def analyze_temporal_consistency(self, current_face):
        """Temporal consistency analysis"""
        if len(self.face_history) < 5:
            return 0.05, ["Insufficient history"]
        
        prev_faces = list(self.face_history)[-5:]
        prev_avg = np.mean(prev_faces, axis=0)
        
        diff = np.mean(np.abs(current_face.astype(float) - prev_avg)) / 255
        
        if diff > 0.20:
            return 0.25, ["Sudden face change (temporal inconsistency)"]
        elif diff > 0.10:
            return 0.10, ["Minor face changes"]
        else:
            return 0.02, ["Consistent"]
    
    def analyze_face_quality(self, face):
        """Complete face quality analysis"""
        if face is None:
            return 0.5, ["No face detected"]
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        blur_score, blur_reason = self.analyze_blur(gray)
        symmetry_score, symmetry_reason = self.analyze_symmetry(gray)
        edge_score, edge_reason = self.analyze_edges(gray)
        color_score, color_reasons = self.analyze_color(face)
        noise_score, noise_reason = self.analyze_noise(gray)
        eye_score, eye_reasons = self.analyze_eye_region(gray)
        temporal_score, temporal_reasons = self.analyze_temporal_consistency(face)
        
        total_score = (
            blur_score * 0.20 +
            symmetry_score * 0.25 +
            edge_score * 0.15 +
            color_score * 0.15 +
            noise_score * 0.10 +
            eye_score * 0.10 +
            temporal_score * 0.05
        )
        
        all_reasons = []
        if blur_score > 0.10:
            all_reasons.append(blur_reason)
        if symmetry_score > 0.10:
            all_reasons.append(symmetry_reason)
        if edge_score > 0.10:
            all_reasons.append(edge_reason)
        all_reasons.extend(color_reasons)
        if noise_score > 0.10:
            all_reasons.append(noise_reason)
        all_reasons.extend(eye_reasons)
        all_reasons.extend(temporal_reasons)
        
        confidence = 1 - total_score
        
        return min(total_score, 0.95), all_reasons[:5], confidence
    
    def analyze_frame(self, frame):
        """Analyze single frame"""
        face = self.extract_faces(frame)
        
        if face is not None:
            fake_prob, reasons, confidence = self.analyze_face_quality(face)
            model_used = "Enhanced Face Analysis"
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:
                fake_prob = 0.55
            elif laplacian_var < 200:
                fake_prob = 0.45
            else:
                fake_prob = 0.35
            
            confidence = abs(fake_prob - 0.5) * 2
            reasons = ["No face detected", "Using full frame analysis"]
            model_used = "Full Frame Analysis"
        
        return {
            'fake_probability': float(fake_prob),
            'real_probability': float(1 - fake_prob),
            'confidence': float(min(confidence, 0.95)),
            'model_used': model_used,
            'reasons': reasons
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
            all_reasons.append("High temporal variance")
        
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


# Mock detector for fallback mode
class MockDetector:
    def __init__(self):
        pass
    
    def analyze_frame(self, frame):
        return {
            'fake_probability': 0.5,
            'real_probability': 0.5,
            'confidence': 0.5,
            'model_used': 'Mock (Fallback Mode)',
            'reasons': ['AI model unavailable', 'Running in demo mode']
        }
    
    def extract_faces(self, frame):
        return None
    
    def analyze_sequence(self, frames_sequence, fps=30):
        return None