import atexit
import os
import glob
import re
import cv2
import math
import joblib
import numpy as np
import mediapipe as mp
import winsound
import threading
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# ALERT AGENT
# ============================================================

from .alert_agent import AlertResponseAgent


# ============================================================
# PROJECT PATHS
# ============================================================

MODEL_PATH = PROJECT_ROOT / "models" / "fall_detection_model.pkl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

OUTPUT_DEMO_PATH = (
    OUTPUTS_DIR / f"fall_detection_session_{SESSION_TIMESTAMP}.mp4"
)

STOP_REQUEST_PATH = OUTPUTS_DIR / "stop_monitoring.flag"

# Live frame used by Flask dashboard
LIVE_FRAME_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "live_frame.jpg"
)

LIVE_FRAME_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("Machine learning model loaded successfully.")


# ============================================================
# 2. INITIALIZE MEDIAPIPE POSE
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
# SEQUENCE CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30

frame_buffer = []


# ============================================================
# PERSISTENT FALL STATE
# ============================================================

fall_latched = False
fall_trigger_counter = 0
head_y_history = []


# ============================================================
# ALERT AGENT STATE
# ============================================================

alert_agent = AlertResponseAgent()

# Prevent repeated alerts while the same fall remains active.
alert_sent_for_current_fall = False


# ============================================================
# ASYNCHRONOUS AUDIO ALARM
# ============================================================

alarm_playing = False


def play_alarm():

    global alarm_playing

    alarm_playing = True

    try:

        winsound.Beep(1200, 500)

    finally:

        alarm_playing = False


def trigger_alarm():

    global alarm_playing

    if not alarm_playing:

        threading.Thread(
            target=play_alarm,
            daemon=True
        ).start()


# ============================================================
# NATURAL SORT
# ============================================================

def natural_sort_key(s):

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(p1, p2):
    """
    Calculates angle of vector p1 -> p2
    relative to vertical axis.

    0 degrees  = standing
    90 degrees = lying down
    """

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.degrees(
        math.atan2(
            abs(dx),
            abs(dy) + 1e-6
        )
    )


# ============================================================
# INPUT PATH CONFIGURATION
# ============================================================

# Webcam
TEST_PATH = 0

# Set TEST_PATH to a video filename or image-sequence directory when testing.


# ============================================================
# IMAGE SEQUENCE LOADER
# ============================================================

image_files = []


if isinstance(TEST_PATH, str) and os.path.isdir(TEST_PATH):

    for ext in (
        "*.png",
        "*.PNG",
        "*.jpg",
        "*.JPG"
    ):

        image_files.extend(
            glob.glob(
                os.path.join(
                    TEST_PATH,
                    "**",
                    ext
                ),
                recursive=True
            )
        )

    image_files.sort(
        key=natural_sort_key
    )


# ============================================================
# VIDEO / WEBCAM LOADER
# ============================================================

cap = None


if not image_files:

    cap = cv2.VideoCapture(TEST_PATH)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video source: {TEST_PATH}"
        )


# ============================================================
# DEMO RECORDING
# ============================================================

SAVE_OUTPUT_DEMO = True
RECORDING_FPS = 25

output_writer = None
cleanup_completed = False


def init_writer(output_path, width, height):

    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.CAP_MSMF,
        fourcc,
        RECORDING_FPS,
        (width, height)
    )

    if not writer.isOpened():
        print(f"Could not open video writer: {output_path}")
        writer.release()
        return None

    return writer


def finalize_recording(writer, output_path):

    writer.release()

    if output_path.exists():
        print(f"Video recording finalized: {output_path}")
    else:
        print(f"Video recording file was not created: {output_path}")


def cleanup_recordings():
    global cap, output_writer, cleanup_completed

    if cleanup_completed:
        return

    cleanup_completed = True

    if cap is not None:
        cap.release()
        cap = None

    if output_writer is not None:
        finalize_recording(
            output_writer,
            OUTPUT_DEMO_PATH
        )
        output_writer = None

    if STOP_REQUEST_PATH.exists():
        STOP_REQUEST_PATH.unlink()

    pose.close()
    cv2.destroyAllWindows()


atexit.register(cleanup_recordings)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("FallDetection.AI")
print("Real-Time Fall Detection + Alert Response Agent")
print("=" * 60)

print(
    "Starting real-time fall detection inference loop..."
)

print("Camera is running through the Flask dashboard.")
print("Use the dashboard STOP button to stop monitoring.")


frame_idx = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    if STOP_REQUEST_PATH.exists():
        print("Stop request received; finalizing video recordings.")
        break

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    if image_files:

        if frame_idx >= len(image_files):

            print(
                "Sequence folder end reached."
            )

            break

        frame = cv2.imread(
            image_files[frame_idx]
        )

        frame_idx += 1

    else:

        ret, frame = cap.read()

        if not ret:

            print(
                "Video stream end reached."
            )

            break


    # --------------------------------------------------------
    # FRAME VALIDATION
    # --------------------------------------------------------

    if frame is None:
        continue


    # --------------------------------------------------------
    # INITIALIZE OUTPUT WRITER
    # --------------------------------------------------------

    if SAVE_OUTPUT_DEMO and output_writer is None:

        h, w, _ = frame.shape
        output_writer = init_writer(
            OUTPUT_DEMO_PATH,
            w,
            h
        )


    # --------------------------------------------------------
    # MEDIAPIPE PROCESSING
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(
        rgb_frame
    )


    # ========================================================
    # PERSON DETECTED
    # ========================================================

    if results.pose_landmarks:

        # ----------------------------------------------------
        # DRAW SKELETON
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lms = results.pose_landmarks.landmark


        # ----------------------------------------------------
        # EXTRACT SHOULDER POINTS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # EXTRACT HIP POINTS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CALCULATE BODY CENTER
        # ----------------------------------------------------

        s_mid = [
            (left_s[0] + right_s[0]) / 2.0,
            (left_s[1] + right_s[1]) / 2.0
        ]


        h_mid = [
            (left_h[0] + right_h[0]) / 2.0,
            (left_h[1] + right_h[1]) / 2.0
        ]


        # ----------------------------------------------------
        # TORSO ANGLE
        # ----------------------------------------------------

        torso_angle = calculate_angle(
            s_mid,
            h_mid
        )


        # ----------------------------------------------------
        # NOSE POSITION
        # ----------------------------------------------------

        nose_y = lms[
            mp_pose.PoseLandmark.NOSE
        ].y


        # ====================================================
        # TRACK DOWNWARD HEAD VELOCITY
        # ====================================================

        head_y_history.append(
            nose_y
        )

        if len(head_y_history) > 10:

            head_y_history.pop(0)


        if len(head_y_history) > 1:

            downward_velocity = (
                head_y_history[-1]
                -
                head_y_history[0]
            )

        else:

            downward_velocity = 0.0


        # ====================================================
        # CONSTRUCT ML FEATURE VECTOR
        # ====================================================

        landmarks = []

        for lm in lms:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])


        frame_buffer.append(
            landmarks
        )


        # ----------------------------------------------------
        # MACHINE LEARNING FALL PROBABILITY
        # ----------------------------------------------------

        ml_fall_prob = 0.0


        if len(frame_buffer) == SEQUENCE_LENGTH:

            input_data = (
                np.array(frame_buffer)
                .flatten()
                .reshape(1, -1)
            )


            ml_fall_prob = (
                model.predict_proba(
                    input_data
                )[0][1]
            )


            frame_buffer.pop(0)


        # ====================================================
        # FALL DETECTION STATE MACHINE
        # ====================================================

        is_falling_motion = (
            downward_velocity > 0.12
            or
            torso_angle > 50.0
            or
            ml_fall_prob > 0.50
        )


        if is_falling_motion:

            fall_trigger_counter += 1

            if fall_trigger_counter >= 2:

                fall_latched = True


        # ====================================================
        # STANDING / RECOVERY DETECTION
        # ====================================================

        is_standing_upright = (
            torso_angle < 30.0
            and
            nose_y < 0.50
        )


        if is_standing_upright:

            fall_latched = False
            fall_trigger_counter = 0


    # ========================================================
    # UI + ALERT AGENT
    # ========================================================

    if fall_latched:

        label_text = (
            "FALL DETECTED! (LATCHED ALARM)"
        )

        color = (0, 0, 255)


        # ----------------------------------------------------
        # AUDIO ALARM
        # ----------------------------------------------------

        trigger_alarm()


        # ----------------------------------------------------
        # ALERT RESPONSE AGENT
        # ----------------------------------------------------

        if not alert_sent_for_current_fall:

            print("\n")
            print("=" * 60)
            print(
                "ALERT RESPONSE AGENT ACTIVATED"
            )
            print("=" * 60)


            
            alert_message = (
                alert_agent.create_alert(
                    camera_name="Laptop Camera",
                    confidence=ml_fall_prob
                )
            )

            print(alert_message)


            print("=" * 60)
            print(
                "ALERT GENERATED SUCCESSFULLY"
            )
            print("=" * 60)
            print("\n")


            # Prevent repeated alerts
            # for the same fall.

            alert_sent_for_current_fall = True


    else:

        label_text = "Normal Activity"

        color = (0, 255, 0)


        # ----------------------------------------------------
        # RESET ALERT STATE
        # ----------------------------------------------------

        # When person becomes upright again,
        # next fall can generate a new alert.

        alert_sent_for_current_fall = False


    # ========================================================
    # DRAW STATUS BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (680, 75),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        label_text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2
    )


    # ========================================================
    # SAVE LIVE FRAME FOR WEB DASHBOARD
    # ========================================================

    try:

        cv2.imwrite(
            str(LIVE_FRAME_PATH),
            frame
        )

    except Exception as error:

        print(
            "Live frame save error:",
            error
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    # OpenCV display window is intentionally disabled.
    # The processed frame is saved to LIVE_FRAME_PATH above
    # and displayed inside the Flask browser dashboard.


    # ========================================================
    # SAVE DEMO VIDEO
    # ========================================================

    if output_writer is not None:

        output_writer.write(
            frame
        )


    # ========================================================
    # EXIT
    # ========================================================

    # Exit is controlled by the Flask dashboard STOP button.
    # No cv2.waitKey() is used because no OpenCV window is opened.


# ============================================================
# CLEANUP
# ============================================================

cleanup_recordings()


print("=" * 60)
print("Execution finished.")
print(
    f"Output saved to: {OUTPUT_DEMO_PATH}"
)
print("=" * 60)