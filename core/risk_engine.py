"""
Risk Engine Module
Calculates unified risk score from multiple detection factors
"""

class RiskEngine:
    def __init__(self):
        self.weights = {
            'face': 0.40,
            'audio': 0.25,
            'temporal': 0.20,
            'lip_sync': 0.15
        }
    
    def calculate_unified_risk(self, face_score, audio_score, temporal_score, lip_sync_score=0.5):
        """Calculate unified risk score (0-100)"""
        total = (
            face_score * self.weights['face'] +
            audio_score * self.weights['audio'] +
            temporal_score * self.weights['temporal'] +
            lip_sync_score * self.weights['lip_sync']
        )
        return total * 100
    
    def get_risk_level(self, unified_score):
        """Get risk level based on unified score"""
        if unified_score > 60:
            return "HIGH", "🔴"
        elif unified_score > 30:
            return "MEDIUM", "🟡"
        else:
            return "LOW", "🟢"