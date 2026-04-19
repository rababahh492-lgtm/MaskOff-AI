import os
import numpy as np
import subprocess

class AudioAnalyzer:
    def __init__(self):
        self.available = False
        self.has_ffmpeg = self.check_ffmpeg()
        
        try:
            import librosa
            self.librosa = librosa
            self.available = True
            print("✅ Advanced audio analysis ready (librosa)")
        except ImportError:
            print("⚠️ Librosa not installed. Using simple fallback.")
    
    def check_ffmpeg(self):
        """التحقق من وجود ffmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def extract_audio_from_video(self, video_path):
        """استخراج الصوت من الفيديو"""
        if not self.has_ffmpeg:
            return None
        
        audio_path = video_path.replace('.mp4', '_temp_audio.wav')
        
        try:
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                   '-ar', '16000', '-ac', '1', '-y', audio_path]
            subprocess.run(cmd, capture_output=True, timeout=30)
            return audio_path if os.path.exists(audio_path) else None
        except:
            return None
    
    def extract_simple_features(self, audio_path):
        """استخراج ميزات بسيطة بدون librosa (fallback)"""
        try:
            import wave
            import struct
            
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                samples = struct.unpack(f'{len(frames)//2}h', frames)
                
                mean_amplitude = np.mean(np.abs(samples))
                std_amplitude = np.std(samples)
                
                fake_score = 0.3
                reasons = []
                
                if std_amplitude < 500:
                    fake_score += 0.2
                    reasons.append("Unusual amplitude pattern detected")
                else:
                    reasons.append("Normal amplitude pattern")
                
                if mean_amplitude < 100:
                    fake_score += 0.1
                    reasons.append("Very low audio energy")
                
                return min(fake_score, 0.8), reasons
        except:
            return 0.5, ["Audio analysis requires librosa or proper audio format"]
    
    def analyze_audio_deepfake(self, audio_path):
        """تحليل متقدم للصوت باستخدام MFCC"""
        if not audio_path:
            return None
        
        # إذا كان librosa غير متاح، استخدم fallback
        if not self.available:
            return self.extract_simple_features(audio_path)
        
        try:
            y, sr = self.librosa.load(audio_path, duration=10)
            
            if len(y) == 0:
                return self.extract_simple_features(audio_path)
            
            reasons = []
            fake_score = 0.2
            
            # 1. تحليل MFCC
            mfcc = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.std(mfcc)
            
            if mfcc_std > 60:
                fake_score += 0.25
                reasons.append("Abnormal MFCC variance (voice cloning sign)")
            elif mfcc_std < 15:
                fake_score += 0.15
                reasons.append("Too uniform MFCC (synthetic voice)")
            else:
                reasons.append("Normal MFCC pattern")
            
            # 2. تحليل Spectral Centroid
            spectral_centroid = np.mean(self.librosa.feature.spectral_centroid(y=y, sr=sr))
            
            if spectral_centroid > 3500:
                fake_score += 0.15
                reasons.append("Unusual high frequency pattern")
            elif spectral_centroid < 400:
                fake_score += 0.10
                reasons.append("Unusual low frequency pattern")
            else:
                reasons.append("Normal frequency range")
            
            # 3. تحليل Zero Crossing Rate
            zcr = np.mean(self.librosa.feature.zero_crossing_rate(y))
            
            if zcr > 0.12:
                fake_score += 0.10
                reasons.append("High zero crossing rate (artifacts)")
            else:
                reasons.append("Normal speech pattern")
            
            # 4. تحليل RMS Energy
            rms = np.mean(self.librosa.feature.rms(y=y))
            
            if rms < 0.01:
                fake_score += 0.10
                reasons.append("Very low energy (possible synthesis)")
            
            confidence = abs(fake_score - 0.5) * 2
            fake_score = min(fake_score, 0.95)
            
            return fake_score, reasons[:4]
            
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return self.extract_simple_features(audio_path)
    
    def detect_deepfake_audio(self, video_path):
        """كشف deepfake من الصوت"""
        audio_path = self.extract_audio_from_video(video_path)
        
        if audio_path:
            fake_score, reasons = self.analyze_audio_deepfake(audio_path)
            try:
                os.remove(audio_path)
            except:
                pass
            
            return {
                'fake_probability': float(fake_score),
                'real_probability': float(1 - fake_score),
                'confidence': abs(fake_score - 0.5) * 2,
                'model_used': 'Librosa MFCC + Spectral Analysis' if self.available else 'Simple Fallback',
                'reasons': reasons
            }
        
        return {
            'fake_probability': 0.5,
            'real_probability': 0.5,
            'confidence': 0.3,
            'model_used': 'Audio Analysis Unavailable',
            'reasons': ['No audio track found', 'Install: pip install librosa', 'Or ensure FFmpeg is installed']
        }
