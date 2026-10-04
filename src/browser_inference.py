import cv2
import joblib
import math
import numpy as np
import mediapipe as mp
import threading
from pathlib import Path

from .alert_agent import AlertResponseAgent


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fall_detection_model.pkl"


# ============================================================
# MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("Browser inference ML model loaded successfully.")

fall_class_indices = np.flatnonzero(
    np.asarray(model.classes_) == 1
)

if len(fall_class_indices) != 1:
    raise ValueError(
        "The fall-detection model must contain exactly one class labelled 1."
    )

FALL_CLASS_INDEX = int(fall_class_indices[0])

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 30
MODEL_PREDICTION_INTERVAL = 2

frame_buffer = []
head_y_history = []

fall_latched = False
fall_trigger_counter = 0
heuristic_trigger_counter = 0
alert_sent_for_current_fall = False
frames_since_model_prediction = 0
model_prediction_count = 0
last_ml_fall_prob = 0.0
last_logged_buffer_size = 0

alert_agent = AlertResponseAgent()

state_lock = threading.Lock()


# ============================================================
# ANGLE
# ============================================================

def calculate_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.degrees(
        math.atan2(
            abs(dx),
            abs(dy) + 1e-6
        )
    )


# ============================================================
# PROCESS FRAME
# ============================================================

def _send_alert_in_background(confidence):
    try:
        alert_agent.create_alert(
            camera_name="Browser Camera",
            confidence=confidence
        )

        alerts = alert_agent.get_alert_history()
        telegram_status = (
            alerts[-1].get("telegram_status", "FAILED")
            if alerts
            else "FAILED"
        )

        if telegram_status == "SENT":
            print("[ALERT] Telegram success")
        else:
            print(
                f"[ALERT] Telegram {telegram_status.lower()}"
            )

    except Exception as error:
        print(
            "[ALERT] Alert agent error:",
            error
        )


def process_frame(frame, include_frame=True):

    global fall_latched
    global fall_trigger_counter
    global heuristic_trigger_counter
    global alert_sent_for_current_fall
    global frames_since_model_prediction
    global model_prediction_count
    global last_ml_fall_prob
    global last_logged_buffer_size

    if frame is None:
        return {
            "success": False,
            "detection": "NO FRAME",
            "person_detected": False,
            "confidence": 0.0,
            "alert_triggered": False,
            "landmarks": []
        }

    with state_lock:

        # ====================================================
        # CONVERT BGR → RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        ml_fall_prob = (
            last_ml_fall_prob
            if len(frame_buffer) >= SEQUENCE_LENGTH
            else 0.0
        )
        person_detected = False

        torso_angle = 0.0
        downward_velocity = 0.0

        landmarks_for_browser = []
        model_prediction_updated = False
        fall_candidate = False
        alert_triggered = False


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if results.pose_landmarks:

            person_detected = True

            lms = results.pose_landmarks.landmark


            # ------------------------------------------------
            # SEND LANDMARKS TO BROWSER
            # ------------------------------------------------

            for lm in lms:

                landmarks_for_browser.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility)
                })


            # ------------------------------------------------
            # SHOULDERS
            # ------------------------------------------------

            left_s = [
                lms[
                    mp_pose.PoseLandmark.LEFT_SHOULDER
                ].x,

                lms[
                    mp_pose.PoseLandmark.LEFT_SHOULDER
                ].y
            ]

            right_s = [
                lms[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER
                ].x,

                lms[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER
                ].y
            ]


            # ------------------------------------------------
            # HIPS
            # ------------------------------------------------

            left_h = [
                lms[
                    mp_pose.PoseLandmark.LEFT_HIP
                ].x,

                lms[
                    mp_pose.PoseLandmark.LEFT_HIP
                ].y
            ]

            right_h = [
                lms[
                    mp_pose.PoseLandmark.RIGHT_HIP
                ].x,

                lms[
                    mp_pose.PoseLandmark.RIGHT_HIP
                ].y
            ]


            # ------------------------------------------------
            # BODY CENTER
            # ------------------------------------------------

            s_mid = [
                (left_s[0] + right_s[0]) / 2.0,
                (left_s[1] + right_s[1]) / 2.0
            ]

            h_mid = [
                (left_h[0] + right_h[0]) / 2.0,
                (left_h[1] + right_h[1]) / 2.0
            ]


            # ------------------------------------------------
            # TORSO ANGLE
            # ------------------------------------------------

            torso_angle = calculate_angle(
                s_mid,
                h_mid
            )


            # ------------------------------------------------
            # NOSE
            # ------------------------------------------------

            nose_y = lms[
                mp_pose.PoseLandmark.NOSE
            ].y


            # ------------------------------------------------
            # HEAD VELOCITY
            # ------------------------------------------------

            head_y_history.append(nose_y)

            if len(head_y_history) > 10:
                head_y_history.pop(0)

            if len(head_y_history) > 1:
                downward_velocity = (
                    head_y_history[-1]
                    - head_y_history[0]
                )


            # =================================================
            # ML FEATURES
            # =================================================

            landmarks = []

            for lm in lms:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            frame_buffer.append(landmarks)
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            frames_since_model_prediction += 1

            # =================================================
            # ML PREDICTION
            # =================================================

            ml_ready = (
                len(frame_buffer) >= SEQUENCE_LENGTH
            )

            if (
                ml_ready
                and (
                    model_prediction_count == 0
                    or frames_since_model_prediction
                    >= MODEL_PREDICTION_INTERVAL
                )
            ):

                input_data = (
                    np.asarray(
                        frame_buffer,
                        dtype=np.float32
                    ).reshape(1, -1)
                )

                probabilities = model.predict_proba(
                    input_data
                )[0]

                ml_fall_prob = float(
                    probabilities[FALL_CLASS_INDEX]
                )
                last_ml_fall_prob = ml_fall_prob
                frames_since_model_prediction = 0
                model_prediction_count += 1
                model_prediction_updated = True

            buffer_size = min(
                len(frame_buffer),
                SEQUENCE_LENGTH
            )

            if (
                buffer_size in (10, 20, 25, 30)
                and buffer_size != last_logged_buffer_size
            ):
                print(
                    f"[FRAME] buffer={buffer_size}/{SEQUENCE_LENGTH}"
                )
                last_logged_buffer_size = buffer_size


            # =================================================
            # FALL DETECTION
            # =================================================

            heuristic_fall = (
                downward_velocity > 0.12
                or torso_angle > 50.0
            )

            ml_fall = (
                ml_ready
                and ml_fall_prob > 0.50
            )

            if ml_ready:
                if heuristic_fall:
                    heuristic_trigger_counter += 1
                else:
                    heuristic_trigger_counter = 0

                if model_prediction_updated:
                    if ml_fall:
                        fall_trigger_counter += 1
                    else:
                        fall_trigger_counter = 0

                fall_candidate = bool(
                    heuristic_fall
                    or (
                        model_prediction_updated
                        and ml_fall
                    )
                )

                if (
                    fall_candidate
                    and max(
                        fall_trigger_counter,
                        heuristic_trigger_counter
                    ) <= 2
                ):
                    counter = max(
                        fall_trigger_counter,
                        heuristic_trigger_counter
                    )
                    print(
                        "[FALL] candidate=True "
                        f"counter={counter}"
                    )

                if (
                    fall_trigger_counter >= 2
                    or heuristic_trigger_counter >= 2
                ):
                    fall_latched = True

            if model_prediction_updated:
                if (
                    model_prediction_count == 1
                    or model_prediction_count % 10 == 0
                ):
                    print(
                        "[MODEL] ready=True "
                        f"probability={ml_fall_prob:.2f}"
                    )

                is_standing_upright = (
                    torso_angle < 30.0
                    and nose_y < 0.50
                    and ml_fall_prob < 0.35
                )

                if (
                    fall_latched
                    and not fall_candidate
                    and is_standing_upright
                ):
                    fall_latched = False
                    fall_trigger_counter = 0
                    heuristic_trigger_counter = 0
                    alert_sent_for_current_fall = False
                    print(
                        "[FALL] recovery detected; alert latch reset"
                    )

        else:

            # No person detected
            landmarks_for_browser = []
            if not fall_latched:
                fall_trigger_counter = 0
                heuristic_trigger_counter = 0


        # ====================================================
        # FALL RESULT
        # ====================================================

        if fall_latched:

            detection = "FALL DETECTED"

            if not alert_sent_for_current_fall:
                alert_sent_for_current_fall = True
                alert_triggered = True
                print("[ALERT] Triggering alert")
                threading.Thread(
                    target=_send_alert_in_background,
                    args=(ml_fall_prob,),
                    name="fall-alert",
                    daemon=True
                ).start()

        else:

            detection = "NORMAL ACTIVITY"

            alert_sent_for_current_fall = False


        if include_frame:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=2,
                        circle_radius=3
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 0),
                        thickness=2,
                        circle_radius=2
                    )
                )

            if fall_latched:
                color = (0, 0, 255)
                label = "FALL DETECTED"
            else:
                color = (0, 255, 0)
                label = "NORMAL ACTIVITY"

            cv2.rectangle(
                frame,
                (20, 20),
                (560, 90),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                frame,
                label,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2
            )
            cv2.putText(
                frame,
                f"ML Confidence: {ml_fall_prob * 100:.1f}%",
                (30, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "success": True,

            "detection": detection,

            "person_detected": person_detected,

            "confidence": round(
                ml_fall_prob * 100,
                2
            ),

            "alert_triggered": alert_triggered,

            "fall_candidate": fall_candidate,

            "fall_trigger_counter": fall_trigger_counter,

            "fall_latched": fall_latched,

            "model_prediction_updated": model_prediction_updated,

            "torso_angle": round(
                torso_angle,
                2
            ),

            "downward_velocity": round(
                downward_velocity,
                4
            ),

            "ml_ready": (
                len(frame_buffer) >= SEQUENCE_LENGTH
            ),

            "buffer_size": min(
                len(frame_buffer),
                SEQUENCE_LENGTH
            ),

            "landmarks": landmarks_for_browser,

            "frame": frame if include_frame else None
        }


# ============================================================
# RESET
# ============================================================

def reset_inference():

    global frame_buffer
    global head_y_history
    global fall_latched
    global fall_trigger_counter
    global heuristic_trigger_counter
    global alert_sent_for_current_fall
    global frames_since_model_prediction
    global model_prediction_count
    global last_ml_fall_prob
    global last_logged_buffer_size

    with state_lock:

        frame_buffer = []
        head_y_history = []

        fall_latched = False
        fall_trigger_counter = 0
        heuristic_trigger_counter = 0

        alert_sent_for_current_fall = False
        frames_since_model_prediction = 0
        model_prediction_count = 0
        last_ml_fall_prob = 0.0
        last_logged_buffer_size = 0