"""
Explainability Module
Extracts top reasons for detection verdict
"""

class ExplainabilityEngine:
    def __init__(self):
        pass
    
    def get_top_reasons(self, results, max_reasons=3):
        """Extract top N reasons for the verdict"""
        all_reasons = results.get('reasons', []).copy()
        
        # Add automatic reasons based on results
        if results.get('suspicious_count', 0) > results.get('analyzed_frames', 1) * 0.5:
            all_reasons.append("⚠️ High number of suspicious frames detected")
        
        if results.get('authenticity_score', 100) < 50:
            all_reasons.append("⚠️ Low authenticity score indicates manipulation")
        
        if results.get('authenticity_score', 100) > 80:
            all_reasons.append("✅ High authenticity score - video appears real")
        
        # Add audio reasons
        audio_reasons = results.get('audio_reasons', [])
        all_reasons.extend(audio_reasons)
        
        # Remove duplicates
        seen = set()
        unique_reasons = []
        for r in all_reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)
        
        return unique_reasons[:max_reasons] if unique_reasons else ["Analysis complete", "No major anomalies detected"]
    
    def format_reasons_ui(self, reasons):
        """Format reasons for UI display with emojis"""
        formatted = []
        emojis = ["🔴", "🟡", "🟢", "📌", "⚠️", "✅"]
        
        for i, reason in enumerate(reasons):
            emoji = emojis[i % len(emojis)]
            formatted.append(f"{emoji} {reason}")
        
        return formatted