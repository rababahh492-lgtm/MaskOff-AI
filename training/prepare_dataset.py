import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

class DatasetPreparer:
    def __init__(self):
        self.img_size = (224, 224)
        
    def extract_frames_from_videos(self, video_dir, output_dir, label, num_frames=100):
        """استخراج فريمات من الفيديوهات"""
        os.makedirs(output_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_name in tqdm(videos, desc=f"Extracting {label} frames"):
            video_path = os.path.join(video_dir, video_name)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // num_frames)
            
            frame_idx = 0
            saved = 0
            
            while cap.isOpened() and saved < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % step == 0:
                    face = self.extract_face(frame)
                    if face is not None:
                        face = cv2.resize(face, self.img_size)
                        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.jpg")
                        cv2.imwrite(output_path, face)
                        saved += 1
                
                frame_idx += 1
            
            cap.release()
    
    def extract_face(self, frame):
        """استخراج الوجه من الفريم"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            return face
        return None
    
    def download_faceforensics(self):
        """تعليمات تحميل FaceForensics++"""
        print("=" * 60)
        print("📥 HOW TO DOWNLOAD FaceForensics++ DATASET")
        print("=" * 60)
        print("""
1. Go to: https://github.com/ondyari/FaceForensics
2. Request access to the dataset
3. Download the dataset (around 500GB)
4. Or use a smaller dataset from Kaggle:
   - https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   - https://www.kaggle.com/datasets/sorour/celeb-df-v2
        """)
        print("=" * 60)
    
    def use_kaggle_dataset(self):
        """استخدام داتاست من Kaggle"""
        print("📥 Downloading dataset from Kaggle...")
        print("Run these commands:")
        print("""
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download xhlulu/140k-real-and-fake-faces

# Unzip
unzip 140k-real-and-fake-faces.zip -d data/raw/
        """)

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.download_faceforensics()