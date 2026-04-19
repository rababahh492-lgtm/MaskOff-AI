from fpdf import FPDF
from datetime import datetime
import os

class ForensicReportGenerator:
    def __init__(self):
        pass
    
    def generate_forensic_report(self, results, video_name):
        """توليد تقرير PDF فورينسيك"""
        try:
            os.makedirs("reports", exist_ok=True)
            
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 20, "MASKOFF AI", ln=True, align="C")
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Forensic Analysis Report", ln=True, align="C")
            
            # Report Info
            pdf.set_font("Arial", "", 12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}", ln=True)
            pdf.cell(0, 10, f"Video: {video_name}", ln=True)
            pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            
            # Results
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Forensic Findings", ln=True)
            pdf.set_font("Arial", "", 12)
            
            authenticity = results.get('authenticity_score', 0)
            suspicious = results.get('suspicious_count', 0)
            analyzed = results.get('analyzed_frames', 0)
            
            pdf.cell(0, 10, f"Total Frames Analyzed: {analyzed}", ln=True)
            pdf.cell(0, 10, f"Suspicious Frames: {suspicious}", ln=True)
            pdf.cell(0, 10, f"Authenticity Score: {authenticity:.1f}%", ln=True)
            
            # Verdict
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Forensic Verdict", ln=True)
            pdf.set_font("Arial", "", 12)
            
            if authenticity > 70:
                pdf.cell(0, 10, "VERDICT: LIKELY AUTHENTIC", ln=True)
                pdf.cell(0, 10, "The video shows no significant signs of manipulation.", ln=True)
            elif authenticity > 40:
                pdf.cell(0, 10, "VERDICT: INCONCLUSIVE", ln=True)
                pdf.cell(0, 10, "The video shows some anomalies. Further investigation recommended.", ln=True)
            else:
                pdf.cell(0, 10, "VERDICT: LIKELY DEEPFAKE", ln=True)
                pdf.cell(0, 10, "Strong evidence of AI manipulation detected.", ln=True)
            
            # Signature
            pdf.ln(20)
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 10, "MaskOff AI - Forensic Analysis System", ln=True)
            pdf.cell(0, 10, "This report is generated automatically by AI analysis.", ln=True)
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"reports/forensic_report_{timestamp}.pdf"
            pdf.output(report_path)
            
            print(f"✅ Report saved: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
