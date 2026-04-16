import os
import cv2
from datetime import datetime

class EvidenceSystem:
    def __init__(self, output_dir="evidence"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def save_suspicious_frame(self, frame, frame_number, score, timestamp):
        """Save suspicious frame as evidence"""
        try:
            filename = f"frame_{frame_number:04d}_score_{score:.2f}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None
    
    def save_face_crop(self, face, frame_number, face_id, timestamp):
        """Save extracted face crop"""
        try:
            filename = f"face_frame_{frame_number:04d}_id_{face_id}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, face)
            return filepath
        except Exception as e:
            print(f"Error saving face: {e}")
            return None
    
    def get_evidence_list(self):
        """Get list of all saved evidence files"""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
            return sorted(files)
        except Exception as e:
            return []
    
    def clear_evidence(self):
        """Clear all evidence files"""
        try:
            for f in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, f)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            return True
        except Exception as e:
            return False