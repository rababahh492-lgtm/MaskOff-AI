"""
Fallback Module
Handles safe mode when dependencies fail
"""

import streamlit as st

class SafeModeManager:
    def __init__(self):
        self.safe_mode = False
        self.errors = []
    
    def check_dependencies(self):
        """Check if all dependencies are available"""
        missing = []
        
        # Check OpenCV
        try:
            import cv2
        except ImportError:
            missing.append("OpenCV (cv2)")
        
        # Check numpy
        try:
            import numpy
        except ImportError:
            missing.append("NumPy")
        
        if missing:
            self.safe_mode = True
            self.errors = missing
            st.warning(f"⚠️ Safe Mode: Missing dependencies - {', '.join(missing)}")
            st.info("💡 Run: pip install -r requirements.txt")
        
        return not self.safe_mode
    
    def get_fallback_detector(self):
        """Return a mock detector when in safe mode"""
        if self.safe_mode:
            from core.detector import MockDetector
            return MockDetector()
        return None