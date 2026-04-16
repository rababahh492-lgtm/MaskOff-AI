# أنشئ ملف download_test_video.py
import urllib.request

def download_test_video():
    """تحميل فيديو تجريبي مع صوت"""
    url = "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"
    output_path = "uploads/test_video_with_audio.mp4"
    
    print("📥 Downloading test video...")
    urllib.request.urlretrieve(url, output_path)
    print(f"✅ Video saved to: {output_path}")

if __name__ == "__main__":
    download_test_video()