# App-Posture-for-you

import cv2
import mediapipe as mp
import math
import time
import threading
import os
from flask import Flask, Response
from playsound import playsound

# ======================
# ĐƯỜNG DẪN ÂM THANH (CHẠY ĐƯỢC KHI ĐÓNG EXE)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_BAD = os.path.join(BASE_DIR, "audio_stoop.mp3")
AUDIO_GOOD = os.path.join(BASE_DIR, "audio_posture.mp3")

# ======================
# PHÁT ÂM THANH (KHÔNG CHỒNG)
# ======================
current_sound = None
sound_lock = threading.Lock()

def play_once(path):
    global current_sound
    with sound_lock:
        if current_sound == path:
            return
        current_sound = path
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

# ======================
# TOÁN HỌC
# ======================
def midpoint(A, B):
    return ((A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2)

def vector(A, B):
    return (A[0] - B[0], A[1] - B[1], A[2] - B[2])

def angle_between(u, v):
    dot = sum(u[i]*v[i] for i in range(3))
    mag_u = math.sqrt(sum(i*i for i in u))
    mag_v = math.sqrt(sum(i*i for i in v))
    if mag_u * mag_v == 0:
        return 180
    cos_theta = max(-1, min(1, dot / (mag_u * mag_v)))
    return int(math.degrees(math.acos(cos_theta)))

# ======================
# MEDIAPIPE
# ======================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================
# FLASK STREAM
# ======================
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    print("🌐 Xem camera tại: http://<IP>:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)
red_line_y = None
prev_time = 0
current_state = None  # "GOOD" | "BAD"

# ======================
# MAIN LOOP
# ======================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    cv2.putText(frame, f"FPS: {fps}", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    if red_line_y is None:
        cv2.putText(frame, 'Click "O" to set redline',
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,255), 2)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

        lm = result.pose_landmarks.landmark

        # Landmark
        LE, RE = 7, 8
        LS, RS = 11, 12
        LH, RH = 23, 24

        ear_mid = midpoint(lm[LE], lm[RE])
        shoulder_mid = midpoint(lm[LS], lm[RS])
        hip_mid = midpoint(lm[LH], lm[RH])

        vec1 = vector(ear_mid, shoulder_mid)
        vec2 = vector(hip_mid, shoulder_mid)

        angle = angle_between(vec1, vec2)

        cv2.putText(frame, f"Angle: {angle}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        # Eye
        left_eye_y = int(lm[2].y * h)
        right_eye_y = int(lm[5].y * h)
        avg_eye_y = int((left_eye_y + right_eye_y) / 2)

        cv2.circle(frame, (int(lm[2].x*w), left_eye_y), 6, (255,0,0), -1)
        cv2.circle(frame, (int(lm[5].x*w), right_eye_y), 6, (0,255,255), -1)

        cv2.putText(frame, f"L:{left_eye_y} R:{right_eye_y}",
                    (w - 260, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if cv2.waitKey(1) & 0xFF == ord('o'):
            red_line_y = avg_eye_y + 17

        if red_line_y:
            cv2.line(frame, (0, red_line_y), (w, red_line_y),
                     (0,0,255), 2)

        bad = angle < 165 or (red_line_y and avg_eye_y > red_line_y + 10)

        if bad:
            if current_state != "BAD":
                play_once(AUDIO_BAD)
                current_state = "BAD"
            cv2.putText(frame, "BAD POSTURE", (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            if current_state != "GOOD":
                play_once(AUDIO_GOOD)
                current_state = "GOOD"
            cv2.putText(frame, "POSTURE OK", (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Posture Monitor", frame)

    with lock:
        output_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

