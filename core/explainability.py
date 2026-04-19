class ExplainabilityEngine:
    def __init__(self):
        pass
    
    def get_top_reasons(self, results, max_reasons=3):
        
        all_reasons = results.get('reasons', []).copy()
        
        
        if results.get('suspicious_count', 0) > results.get('analyzed_frames', 1) * 0.5:
            all_reasons.append("⚠️ High number of suspicious frames detected")
        
        if results.get('authenticity_score', 100) < 50:
            all_reasons.append("⚠️ Low authenticity score indicates manipulation")
        
        if results.get('authenticity_score', 100) > 80:
            all_reasons.append("✅ High authenticity score - video appears real")
        
       
        audio_reasons = results.get('audio_reasons', [])
        all_reasons.extend(audio_reasons)
        
        
        seen = set()
        unique_reasons = []
        for r in all_reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)
        
        return unique_reasons[:max_reasons] if unique_reasons else ["Analysis complete", "No major anomalies detected"]
    
    def get_top_reasons_with_weights(self, results):
        
        reasons_with_weights = []
        
       
        if results.get('temporal_reason'):
            reasons_with_weights.append({
                'reason': results['temporal_reason'],
                'weight': 0.8
            })
        
       
        suspicious_count = results.get('suspicious_count', 0)
        analyzed_frames = results.get('analyzed_frames', 1)
        if suspicious_count > analyzed_frames * 0.5:
            reasons_with_weights.append({
                'reason': f"High number of suspicious frames ({suspicious_count}/{analyzed_frames})",
                'weight': 0.9
            })
        
        
        authenticity = results.get('authenticity_score', 50)
        if authenticity < 40:
            reasons_with_weights.append({
                'reason': f"Very low authenticity score ({authenticity:.0f}%)",
                'weight': 0.85
            })
        
       
        for reason in results.get('audio_reasons', []):
            weight = 0.7 if 'abnormal' in reason.lower() else 0.4
            reasons_with_weights.append({'reason': reason, 'weight': weight})
        
        
        reasons_with_weights.sort(key=lambda x: x['weight'], reverse=True)
        
        return reasons_with_weights[:3]
    
    def format_reasons_ui(self, reasons):
        
        formatted = []
        emojis = ["🔴", "🟡", "🟢", "📌", "⚠️", "✅"]
        
        for i, reason in enumerate(reasons):
            emoji = emojis[i % len(emojis)]
            formatted.append(f"{emoji} {reason}")
        
        return formatted
