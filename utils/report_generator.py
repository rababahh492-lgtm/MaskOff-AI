from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # لمنع مشاكل matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class ForensicReportGenerator:
    def __init__(self):
        self.pdf = None
    
    def add_header(self, pdf, title):
        """إضافة رأس الصفحة"""
        pdf.set_font("Arial", "B", 24)
        pdf.set_text_color(102, 126, 234)
        pdf.cell(0, 20, "MASKOFF AI", ln=True, align="C")
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(10)
    
    def add_risk_graph(self, pdf, frame_analyses):
        """إضافة رسم بياني للمخاطر"""
        if not frame_analyses:
            pdf.cell(0, 10, "No frame analysis data available", ln=True)
            return
        
        try:
            # إنشاء مجلد مؤقت للصور
            os.makedirs("temp", exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            frames = [f['frame_num'] for f in frame_analyses]
            risks = [f['fake_prob'] for f in frame_analyses]
            
            plt.plot(frames, risks, color='#ff4757', linewidth=2, label='Deepfake Probability')
            plt.axhline(y=0.5, color='#ffa502', linestyle='--', linewidth=2, label='Threshold (50%)')
            plt.fill_between(frames, risks, 0.5, 
                             where=(np.array(risks) > 0.5), 
                             color='#ff4757', alpha=0.3, label='Suspicious Region')
            plt.fill_between(frames, risks, 0.5, 
                             where=(np.array(risks) <= 0.5), 
                             color='#2ed573', alpha=0.3, label='Safe Region')
            
            plt.xlabel('Frame Number', fontsize=12)
            plt.ylabel('Deepfake Probability', fontsize=12)
            plt.title('Frame-by-Frame Deepfake Risk Analysis', fontsize=14, fontweight='bold')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            img_path = 'temp/risk_graph.png'
            plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            if os.path.exists(img_path):
                pdf.image(img_path, x=10, y=None, w=190)
                os.remove(img_path)
        except Exception as e:
            pdf.cell(0, 10, f"Error generating graph: {str(e)[:50]}", ln=True)
    
    def add_risk_summary_table(self, pdf, results):
        """إضافة جدول ملخص المخاطر"""
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Risk Summary", ln=True)
        pdf.ln(5)
        
        # عناوين الجدول
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(90, 10, "Metric", 1, 0, 'C', 1)
        pdf.cell(90, 10, "Value", 1, 1, 'C', 1)
        
        # بيانات الجدول
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        
        authenticity = results.get('authenticity_score', 0)
        
        metrics = [
            ("Total Frames Analyzed", str(results.get('analyzed_frames', 0))),
            ("Suspicious Frames", str(results.get('suspicious_count', 0))),
            ("Authenticity Score", f"{authenticity:.1f}%"),
            ("Model Used", results.get('model_used', 'Face Analysis')),
            ("Analysis Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ]
        
        for metric, value in metrics:
            pdf.cell(90, 8, metric, 1)
            pdf.cell(90, 8, value, 1)
            pdf.ln()
    
    def add_verdict_section(self, pdf, results):
        """إضافة قسم الحكم النهائي"""
        authenticity = results.get('authenticity_score', 0)
        
        if authenticity > 70:
            verdict = "LIKELY AUTHENTIC ✓"
            color = (46, 213, 115)
            message = "The video shows no significant signs of manipulation. It appears to be authentic."
        elif authenticity > 40:
            verdict = "INCONCLUSIVE ⚠"
            color = (255, 165, 2)
            message = "The video shows some anomalies. Further investigation is recommended."
        else:
            verdict = "LIKELY DEEPFAKE ✗"
            color = (255, 71, 87)
            message = "The video shows strong signs of AI manipulation. Deepfake detection is highly probable."
        
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(color[0], color[1], color[2])
        pdf.cell(0, 10, f"Verdict: {verdict}", ln=True)
        
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, message)
        pdf.ln(5)
    
    def add_reasons_section(self, pdf, results):
        """إضافة قسم الأسباب والتحليلات"""
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Analysis Details & Detection Factors", ln=True)
        pdf.ln(5)
        
        reasons = results.get('reasons', [])
        
        if reasons:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "The following factors contributed to the analysis:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for i, reason in enumerate(reasons[:10], 1):
                pdf.cell(5, 6, "", ln=0)
                pdf.cell(0, 6, f"{i}. {reason}", ln=True)
        else:
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, "Analysis based on:", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(5, 6, "", ln=0)
            pdf.cell(0, 6, "1. Face quality analysis (blur, symmetry, edges)", ln=True)
            pdf.cell(5, 6, "", ln=0)
            pdf.cell(0, 6, "2. Color and noise analysis", ln=True)
            pdf.cell(5, 6, "", ln=0)
            pdf.cell(0, 6, "3. Temporal consistency", ln=True)
            
            # إضافة أسباب الصوت إذا وجدت
            audio_reasons = results.get('audio_reasons', [])
            if audio_reasons:
                pdf.ln(3)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 8, "Audio Analysis Factors:", ln=True)
                pdf.set_font("Arial", "", 10)
                for i, reason in enumerate(audio_reasons[:5], 1):
                    pdf.cell(5, 6, "", ln=0)
                    pdf.cell(0, 6, f"{i}. {reason}", ln=True)
    
    def add_suspicious_frames_table(self, pdf, suspicious_frames):
        """إضافة جدول الفريمات المشبوهة"""
        if not suspicious_frames:
            pdf.cell(0, 10, "No suspicious frames detected.", ln=True)
            return
        
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Suspicious Frames Details", ln=True)
        pdf.ln(5)
        
        # عناوين الجدول
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(20, 10, "#", 1, 0, 'C', 1)
        pdf.cell(50, 10, "Frame Number", 1, 0, 'C', 1)
        pdf.cell(50, 10, "Risk Score", 1, 0, 'C', 1)
        pdf.cell(60, 10, "Confidence", 1, 1, 'C', 1)
        
        # بيانات الجدول
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        
        for i, frame in enumerate(suspicious_frames[:20], 1):
            pdf.cell(20, 8, str(i), 1, 0, 'C')
            pdf.cell(50, 8, f"#{frame.get('frame_number', 'N/A')}", 1, 0, 'C')
            
            # لون حسب درجة الخطر
            score = frame.get('score', 0)
            if score > 0.7:
                pdf.set_text_color(255, 71, 87)
            elif score > 0.5:
                pdf.set_text_color(255, 165, 2)
            else:
                pdf.set_text_color(46, 213, 115)
            
            pdf.cell(50, 8, f"{score:.1%}", 1, 0, 'C')
            pdf.set_text_color(0, 0, 0)
            pdf.cell(60, 8, f"{frame.get('confidence', 0):.1%}", 1, 1, 'C')
    
    def generate_forensic_report(self, results, video_name):
        """توليد تقرير PDF كامل ومحسن"""
        try:
            # إنشاء مجلد reports
            os.makedirs("reports", exist_ok=True)
            os.makedirs("temp", exist_ok=True)
            
            pdf = FPDF()
            
            # ========== صفحة 1: الغلاف ==========
            pdf.add_page()
            self.add_header(pdf, "Forensic Analysis Report")
            
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}", ln=True)
            pdf.cell(0, 10, f"Video File: {video_name}", ln=True)
            pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            
            # ========== صفحة 2: الملخص التنفيذي ==========
            pdf.add_page()
            self.add_header(pdf, "Executive Summary")
            self.add_verdict_section(pdf, results)
            
            # ========== صفحة 3: جدول المخاطر ==========
            pdf.add_page()
            self.add_header(pdf, "Risk Assessment")
            self.add_risk_summary_table(pdf, results)
            
            # ========== صفحة 4: الرسم البياني ==========
            if results.get('frame_analyses'):
                pdf.add_page()
                self.add_header(pdf, "Risk Timeline")
                self.add_risk_graph(pdf, results['frame_analyses'])
            
            # ========== صفحة 5: أسباب الكشف ==========
            pdf.add_page()
            self.add_header(pdf, "Detection Factors")
            self.add_reasons_section(pdf, results)
            
            # ========== صفحة 6: الفريمات المشبوهة ==========
            suspicious_frames = results.get('suspicious_frames_data', [])
            if suspicious_frames:
                self.add_suspicious_frames_table(pdf, suspicious_frames)
            
            # حفظ التقرير
            report_path = f"reports/forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(report_path)
            
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {e}")
            # محاولة حفظ تقرير بسيط إذا فشل التقرير الكامل
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "MaskOff AI Report", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.cell(0, 10, f"Authenticity Score: {results.get('authenticity_score', 0):.1f}%", ln=True)
                pdf.cell(0, 10, f"Suspicious Frames: {results.get('suspicious_count', 0)}", ln=True)
                
                report_path = f"reports/simple_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(report_path)
                return report_path
            except:
                return None