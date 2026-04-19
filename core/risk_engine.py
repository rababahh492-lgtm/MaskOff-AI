
import numpy as np

class RiskEngine:
    def __init__(self):
        
        self.base_weights = {
            'face': 0.40,
            'audio': 0.25,
            'temporal': 0.20,
            'lip_sync': 0.15
        }
    
    def calculate_dynamic_weights(self, face_quality, audio_quality, temporal_quality, lip_sync_quality=0.5):
    
        
        weights = self.base_weights.copy()
        
        if lip_sync_quality == 0.5:
            weights['lip_sync'] = 0
            weights['face'] += 0.15
        
        if audio_quality < 0.3:
            weights['audio'] = 0.10
            weights['face'] += 0.15
        
       
        if temporal_quality < 0.2:
            weights['temporal'] = 0.10
            weights['face'] += 0.10
        
        if face_quality < 0.3:
            weights['face'] = 0.20
            weights['audio'] += 0.10
            weights['temporal'] += 0.10
        
        
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        
        return weights
    
    def calculate_unified_risk(self, face_score, audio_score, temporal_score, 
                           face_quality=0.5, audio_quality=0.5, temporal_quality=0.5,
                           lip_sync_score=0.5, lip_sync_quality=0.5):
    
     weights = self.calculate_dynamic_weights(face_quality, audio_quality, temporal_quality, lip_sync_quality)
    
     total = (
        face_score * weights['face'] +
        audio_score * weights['audio'] +
        temporal_score * weights['temporal'] +
        lip_sync_score * weights['lip_sync']
     )
    
     return total * 100
    
    def calculate_weighted_confidence(self, face_conf, audio_conf, temporal_conf, lip_sync_conf=0.5):
     
        weights = self.calculate_dynamic_weights(face_conf, audio_conf, temporal_conf, lip_sync_conf)
        
        final_confidence = (
            face_conf * weights['face'] +
            audio_conf * weights['audio'] +
            temporal_conf * weights['temporal']
        )
        
        return min(final_confidence, 0.95)
    
    def get_risk_level(self, unified_score):
        
        if unified_score > 60:
            return "HIGH", "🔴"
        elif unified_score > 30:
            return "MEDIUM", "🟡"
        else:
            return "LOW", "🟢"
    
    def get_risk_description(self, unified_score):
       
        if unified_score > 60:
            return "High probability of deepfake manipulation"
        elif unified_score > 30:
            return "Medium risk - further investigation recommended"
        else:
            return "Low risk - video appears authentic"
    
    def detect_temporal_anomalies(self, fake_probs):
        
        if len(fake_probs) < 5:
            return 0.5, "Insufficient data", 0.3
        
        variance = np.var(fake_probs)
        mean = np.mean(fake_probs)
        std = np.std(fake_probs) + 1e-6
        
        
        spikes = 0
        for i in range(1, len(fake_probs) - 1):
            z = (fake_probs[i] - mean) / std
            if z > 2:
                spikes += 1
        
        spike_ratio = spikes / len(fake_probs)
        
        if spike_ratio > 0.2:
            return 0.85, f"High temporal inconsistency ({spikes} spikes detected)", 0.8
        elif variance > 0.08:
            return 0.65, "Medium temporal variance", 0.6
        else:
            return 0.35, "Stable temporal pattern", 0.7
