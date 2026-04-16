import cv2
import os

# تحميل face detector الجاهز من OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# 1. Extract Frames
# =========================
def extract_frames(video_path, output_folder="frames"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return count


# =========================
# 2. Detect Suspicious Frames (basic motion check)
# =========================
def detect_suspicious_frames(frames_folder="frames", threshold=30):
    frames = sorted(os.listdir(frames_folder))
    suspicious = []

    prev_frame = None

    for frame_name in frames:
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            score = diff.mean()

            if score > threshold:
                suspicious.append((frame_name, score))

        prev_frame = gray

    return suspicious


# =========================
# 3. Extract Faces from Frames
# =========================
def extract_faces_from_frames(frames_folder="frames", output_folder="faces"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = sorted(os.listdir(frames_folder))
    count = 0

    for frame_name in frames:
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_path = os.path.join(output_folder, f"face_{count}.jpg")
            cv2.imwrite(face_path, face)
            count += 1

    return count