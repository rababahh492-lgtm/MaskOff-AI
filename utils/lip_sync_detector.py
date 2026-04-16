import cv2
import numpy as np

class LipSyncDetector:
    def __init__(self):
        # استخدام Haar Cascade المدمج في OpenCV بدلاً من dlib
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def extract_lip_movement(self, frames_sequence):
        """استخراج حركة الشفاه من الفريمات"""
        lip_movements = []
        
        for frame in frames_sequence:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # منطقة الفم (النصف السفلي من الوجه)
                mouth_y = y + int(h * 0.65)
                mouth_h = int(h * 0.25)
                
                if mouth_y + mouth_h <= gray.shape[0]:
                    mouth_region = gray[mouth_y:mouth_y+mouth_h, x:x+w]
                    if mouth_region.size > 0:
                        # حساب فتحة الفم باستخدام التباين
                        mouth_opening = np.std(mouth_region)
                        lip_movements.append(mouth_opening)
                    else:
                        lip_movements.append(0)
                else:
                    lip_movements.append(0)
            else:
                lip_movements.append(0)
        
        return lip_movements
    
    def analyze_lip_sync(self, video_path, audio_path=None):
        """تحليل تطابق حركة الشفاه مع الصوت"""
        
        # استخراج حركة الشفاه من الفيديو
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(100, frame_count)
        step = max(1, frame_count // sample_count)
        
        frame_num = 0
        while len(frames) < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % step == 0:
                frames.append(frame)
            frame_num += 1
        cap.release()
        
        if len(frames) < 5:
            return 0.5, ["Not enough frames for analysis"]
        
        lip_movements = self.extract_lip_movement(frames)
        
        if len(lip_movements) == 0:
            return 0.5, ["No face detected for lip analysis"]
        
        # تحليل حركة الشفاه
        lip_variance = np.var(lip_movements)
        lip_mean = np.mean(lip_movements)
        
        fake_score = 0.2
        reasons = []
        
        # تباين منخفض = حركة شفاه غير طبيعية
        if lip_variance < 30:
            fake_score += 0.4
            reasons.append("⚠️ Very low lip movement - possible deepfake")
        elif lip_variance < 80:
            fake_score += 0.2
            reasons.append("Low lip movement variance")
        elif lip_variance > 300:
            fake_score += 0.1
            reasons.append("Excessive lip movement - unnatural")
        
        # متوسط حركة منخفض
        if lip_mean < 20:
            fake_score += 0.15
            reasons.append("Limited lip movement detected")
        
        fake_score = min(fake_score, 0.95)
        
        if not reasons:
            reasons = ["✓ Normal lip movement detected"]
        
        return fake_score, reasons