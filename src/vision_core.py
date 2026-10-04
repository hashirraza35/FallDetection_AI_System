import collections
import cv2
import joblib
import mediapipe as mp
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fall_detection_model.pkl"

# 1. Load Trained Machine Learning Model
try:
    model = joblib.load(MODEL_PATH)
    print("Successfully loaded 'fall_detection_model.pkl'")
except FileNotFoundError:
    print("Error: 'fall_detection_model.pkl' not found.")
    print("Please run 'python train_from_features.py' first.")
    exit()

# 2. Initialize MediaPipe Pose Engine
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 3. Sliding Window Buffer (30 frames = ~1 second at 30 FPS)
SEQUENCE_LENGTH = 30
frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("AI Fall Detection active! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    status_text = "Buffering sequence..."
    status_color = (255, 255, 0)  # Yellow during buffer warmup

    if results.pose_landmarks:
        # Extract 33 keypoints with (x, y, z) coordinates -> 99 values per frame
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        frame_buffer.append(landmarks)

        # Draw skeleton visualization
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255), thickness=2
            ),
        )

        # 4. Predict State When Buffer Is Full
        if len(frame_buffer) == SEQUENCE_LENGTH:
            # Flatten 30x99 frame array into a 2970-feature row vector
            sequence_vector = np.array(frame_buffer).flatten().reshape(1, -1)

            # Perform Model Inference
            prediction = model.predict(sequence_vector)[0]
            probabilities = model.predict_proba(sequence_vector)[0]

            if prediction == 1:
                confidence = probabilities[1] * 100
                status_text = f"WARNING: FALL DETECTED! ({confidence:.0f}%)"
                status_color = (0, 0, 255)  # Red alert
            else:
                confidence = probabilities[0] * 100
                status_text = f"Status: Normal ({confidence:.0f}%)"
                status_color = (0, 255, 0)  # Green

    # Render detection status on screen
    cv2.putText(
        frame,
        status_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    cv2.imshow("FallDetection.AI - Trained Model Inference Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()