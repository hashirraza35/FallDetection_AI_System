import cv2
import joblib
import math
import numpy as np
import mediapipe as mp
import threading
import time
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

print("Browser inference ML model loaded successfully.", flush=True)

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

# Run ML model every 2 processed frames after buffer is ready.
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
# DIAGNOSTIC CONFIG
# ============================================================

DIAGNOSTIC_ENABLED = True

# Prevent terminal from being flooded.
diagnostic_frame_counter = 0

# Print full timing details for first few frames,
# then every 10th frame.
DIAGNOSTIC_VERBOSE_FRAMES = 5
DIAGNOSTIC_INTERVAL = 10


# ============================================================
# HELPER: DIAGNOSTIC LOG
# ============================================================

def diagnostic_log(message):
    if DIAGNOSTIC_ENABLED:
        print(
            f"[INFERENCE DIAGNOSTIC] {message}",
            flush=True
        )


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
# ALERT
# ============================================================

def _send_alert_in_background(confidence):
    try:
        print(
            f"[ALERT] Starting background alert | confidence={confidence}",
            flush=True
        )

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
            print(
                "[ALERT] Telegram success",
                flush=True
            )
        else:
            print(
                f"[ALERT] Telegram {telegram_status.lower()}",
                flush=True
            )

    except Exception as error:
        print(
            "[ALERT] Alert agent error:",
            error,
            flush=True
        )


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame, include_frame=True):

    global fall_latched
    global fall_trigger_counter
    global heuristic_trigger_counter
    global alert_sent_for_current_fall

    global frames_since_model_prediction
    global model_prediction_count
    global last_ml_fall_prob
    global last_logged_buffer_size

    global diagnostic_frame_counter

    # --------------------------------------------------------
    # FRAME VALIDATION
    # --------------------------------------------------------

    if frame is None:
        return {
            "success": False,
            "detection": "NO FRAME",
            "person_detected": False,
            "confidence": 0.0,
            "alert_triggered": False,
            "landmarks": []
        }

    diagnostic_frame_counter += 1

    current_frame_number = diagnostic_frame_counter

    should_log_diagnostic = (
        current_frame_number <= DIAGNOSTIC_VERBOSE_FRAMES
        or current_frame_number % DIAGNOSTIC_INTERVAL == 0
    )

    total_start = time.perf_counter()

    if should_log_diagnostic:
        diagnostic_log(
            f"FRAME {current_frame_number} | START | "
            f"shape={frame.shape} | "
            f"dtype={frame.dtype}"
        )

    # ========================================================
    # STATE LOCK
    # ========================================================

    with state_lock:

        # ====================================================
        # STEP 1 — BGR → RGB
        # ====================================================

        step_start = time.perf_counter()

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        step_time = time.perf_counter() - step_start

        if should_log_diagnostic:
            diagnostic_log(
                f"FRAME {current_frame_number} | "
                f"STEP 1 cvtColor | {step_time:.4f}s"
            )

        # ====================================================
        # STEP 2 — MEDIAPIPE
        # ====================================================

        step_start = time.perf_counter()

        if should_log_diagnostic:
            diagnostic_log(
                f"FRAME {current_frame_number} | "
                f"STEP 2 pose.process START"
            )

        results = pose.process(rgb_frame)

        step_time = time.perf_counter() - step_start

        if should_log_diagnostic:
            diagnostic_log(
                f"FRAME {current_frame_number} | "
                f"STEP 2 pose.process END | {step_time:.4f}s | "
                f"person={bool(results.pose_landmarks)}"
            )

        # ====================================================
        # INITIAL VALUES
        # ====================================================

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

            if should_log_diagnostic:
                diagnostic_log(
                    f"FRAME {current_frame_number} | "
                    f"PERSON DETECTED | landmarks={len(lms)}"
                )

            # =================================================
            # STEP 3 — LANDMARK SERIALIZATION
            # =================================================

            step_start = time.perf_counter()

            for lm in lms:
                landmarks_for_browser.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility)
                })

            step_time = time.perf_counter() - step_start

            if should_log_diagnostic:
                diagnostic_log(
                    f"FRAME {current_frame_number} | "
                    f"STEP 3 landmarks | {step_time:.4f}s"
                )

            # =================================================
            # SHOULDERS
            # =================================================

            left_shoulder = lms[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]

            right_shoulder = lms[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            left_s = [
                left_shoulder.x,
                left_shoulder.y
            ]

            right_s = [
                right_shoulder.x,
                right_shoulder.y
            ]

            # =================================================
            # HIPS
            # =================================================

            left_hip = lms[
                mp_pose.PoseLandmark.LEFT_HIP
            ]

            right_hip = lms[
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            left_h = [
                left_hip.x,
                left_hip.y
            ]

            right_h = [
                right_hip.x,
                right_hip.y
            ]

            # =================================================
            # BODY CENTER
            # =================================================

            s_mid = [
                (left_s[0] + right_s[0]) / 2.0,
                (left_s[1] + right_s[1]) / 2.0
            ]

            h_mid = [
                (left_h[0] + right_h[0]) / 2.0,
                (left_h[1] + right_h[1]) / 2.0
            ]

            # =================================================
            # TORSO ANGLE
            # =================================================

            torso_angle = calculate_angle(
                s_mid,
                h_mid
            )

            # =================================================
            # NOSE
            # =================================================

            nose_y = lms[
                mp_pose.PoseLandmark.NOSE
            ].y

            # =================================================
            # HEAD VELOCITY
            # =================================================

            head_y_history.append(nose_y)

            if len(head_y_history) > 10:
                head_y_history.pop(0)

            if len(head_y_history) > 1:
                downward_velocity = (
                    head_y_history[-1]
                    - head_y_history[0]
                )

            # =================================================
            # STEP 4 — ML FEATURE EXTRACTION
            # =================================================

            step_start = time.perf_counter()

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

            step_time = time.perf_counter() - step_start

            if should_log_diagnostic:
                diagnostic_log(
                    f"FRAME {current_frame_number} | "
                    f"STEP 4 features | {step_time:.4f}s | "
                    f"buffer={len(frame_buffer)}/{SEQUENCE_LENGTH}"
                )

            # =================================================
            # STEP 5 — ML PREDICTION
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

                if should_log_diagnostic:
                    diagnostic_log(
                        f"FRAME {current_frame_number} | "
                        f"STEP 5 model prediction START | "
                        f"prediction_count={model_prediction_count}"
                    )

                step_start = time.perf_counter()

                input_data = np.asarray(
                    frame_buffer,
                    dtype=np.float32
                ).reshape(1, -1)

                if should_log_diagnostic:
                    diagnostic_log(
                        f"FRAME {current_frame_number} | "
                        f"MODEL INPUT shape={input_data.shape}"
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

                step_time = time.perf_counter() - step_start

                if should_log_diagnostic:
                    diagnostic_log(
                        f"FRAME {current_frame_number} | "
                        f"STEP 5 model prediction END | "
                        f"{step_time:.4f}s | "
                        f"fall_probability={ml_fall_prob:.4f}"
                    )

            buffer_size = min(
                len(frame_buffer),
                SEQUENCE_LENGTH
            )

            if (
                buffer_size in (10, 20, 25, 30)
                and buffer_size != last_logged_buffer_size
            ):

                print(
                    f"[FRAME] buffer={buffer_size}/{SEQUENCE_LENGTH}",
                    flush=True
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
                        f"counter={counter}",
                        flush=True
                    )

                if (
                    fall_trigger_counter >= 2
                    or heuristic_trigger_counter >= 2
                ):
                    fall_latched = True

            # =================================================
            # MODEL LOGGING / RECOVERY
            # =================================================

            if model_prediction_updated:

                if (
                    model_prediction_count == 1
                    or model_prediction_count % 10 == 0
                ):

                    print(
                        "[MODEL] ready=True "
                        f"probability={ml_fall_prob:.2f}",
                        flush=True
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
                        "[FALL] recovery detected; "
                        "alert latch reset",
                        flush=True
                    )

        else:

            # =================================================
            # NO PERSON
            # =================================================

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

                print(
                    "[ALERT] Triggering alert",
                    flush=True
                )

                threading.Thread(
                    target=_send_alert_in_background,
                    args=(ml_fall_prob,),
                    name="fall-alert",
                    daemon=True
                ).start()

        else:

            detection = "NORMAL ACTIVITY"

            alert_sent_for_current_fall = False

        # ====================================================
        # STEP 6 — DRAWING
        # ====================================================

        if include_frame:

            step_start = time.perf_counter()

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

            step_time = time.perf_counter() - step_start

            if should_log_diagnostic:

                diagnostic_log(
                    f"FRAME {current_frame_number} | "
                    f"STEP 6 drawing | {step_time:.4f}s"
                )

        # ====================================================
        # STEP 7 — RETURN
        # ====================================================

        total_time = time.perf_counter() - total_start

        if should_log_diagnostic:

            diagnostic_log(
                f"FRAME {current_frame_number} | "
                f"STEP 7 COMPLETE | "
                f"TOTAL={total_time:.4f}s | "
                f"detection={detection} | "
                f"person={person_detected} | "
                f"buffer={len(frame_buffer)}/{SEQUENCE_LENGTH}"
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

            "model_prediction_updated": (
                model_prediction_updated
            ),

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

            "frame": (
                frame
                if include_frame
                else None
            )
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

    global diagnostic_frame_counter

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

        diagnostic_frame_counter = 0

    print(
        "[INFERENCE] State reset successfully.",
        flush=True
    )