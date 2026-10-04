from flask import Flask, jsonify, send_from_directory, Response, request
from werkzeug.exceptions import RequestEntityTooLarge
from urllib.parse import urlsplit

import os
import time
import json
import cv2
import numpy as np
import threading
import uuid
import hmac

from pathlib import Path

from src.browser_inference import (
    alert_agent,
    process_frame,
    reset_inference
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DASHBOARD_DIR = PROJECT_ROOT / "DASHBOARD"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LIVE_FRAME_PATH = OUTPUTS_DIR / "live_frame.jpg"

ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"
LEGACY_FRAME_INTERVAL_SECONDS = 1.0
MAX_UPLOAD_BYTES = 1_048_576
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
last_legacy_frame_saved_at = 0.0


OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder=DASHBOARD_DIR,
    static_url_path=""
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

REQUIRE_APP_AUTH = os.getenv(
    "APP_REQUIRE_AUTH",
    "false"
).strip().lower() in {"1", "true", "yes"}
APP_ACCESS_USERNAME = os.getenv(
    "APP_ACCESS_USERNAME",
    "admin"
)
APP_ACCESS_PASSWORD = os.getenv(
    "APP_ACCESS_PASSWORD",
    ""
)

if REQUIRE_APP_AUTH and not APP_ACCESS_PASSWORD:
    raise RuntimeError(
        "APP_ACCESS_PASSWORD is required when APP_REQUIRE_AUTH is enabled."
    )


# ============================================================
# PROCESS LOCK
# ============================================================

process_lock = threading.Lock()


# ============================================================
# SESSION STATE
# ============================================================

session_lock = threading.Lock()

current_session_id = None


@app.before_request
def enforce_request_security():
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("Origin")

        if origin:
            parsed_origin = urlsplit(origin)
            forwarded_scheme = request.headers.get(
                "X-Forwarded-Proto",
                request.scheme
            ).split(",", 1)[0].strip().lower()

            if (
                parsed_origin.netloc.casefold()
                != request.host.casefold()
                or parsed_origin.scheme.lower()
                != forwarded_scheme
            ):
                return jsonify({
                    "success": False,
                    "message": "Cross-origin request rejected."
                }), 403

    if not REQUIRE_APP_AUTH or request.path == "/healthz":
        return None

    credentials = request.authorization
    if (
        credentials is None
        or not hmac.compare_digest(
            (credentials.username or "").encode("utf-8"),
            APP_ACCESS_USERNAME.encode("utf-8")
        )
        or not hmac.compare_digest(
            (credentials.password or "").encode("utf-8"),
            APP_ACCESS_PASSWORD.encode("utf-8")
        )
    ):
        return Response(
            "Authentication required.",
            status=401,
            headers={
                "WWW-Authenticate":
                    'Basic realm="FallDetection.AI", charset="UTF-8"'
            },
            content_type="text/plain; charset=utf-8"
        )

    return None


@app.after_request
def add_security_headers(response):
    response.headers.setdefault(
        "X-Content-Type-Options",
        "nosniff"
    )
    response.headers.setdefault(
        "X-Frame-Options",
        "DENY"
    )
    response.headers.setdefault(
        "Referrer-Policy",
        "no-referrer"
    )
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(self), microphone=(), geolocation=()"
    )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )

    if response.mimetype == "application/json":
        response.headers.setdefault(
            "Cache-Control",
            "no-store"
        )

    forwarded_scheme = request.headers.get(
        "X-Forwarded-Proto",
        ""
    ).split(",", 1)[0].strip().lower()

    if request.is_secure or forwarded_scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains"
        )

    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return jsonify({
        "success": False,
        "message": "Request exceeds the 1 MiB upload limit."
    }), 413


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "message": "API endpoint not found."
        }), 404

    return Response(
        "Not found.",
        status=404,
        content_type="text/plain; charset=utf-8"
    )


@app.route("/healthz")
def health_check():
    return jsonify({"status": "ok"})


def get_jpeg_dimensions(image_bytes):
    if len(image_bytes) < 4 or image_bytes[:2] != b"\xff\xd8":
        return None

    frame_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF
    }
    standalone_markers = {
        0x01, 0xD8, 0xD9, 0xD0, 0xD1, 0xD2, 0xD3,
        0xD4, 0xD5, 0xD6, 0xD7
    }
    position = 2

    while position < len(image_bytes):
        if image_bytes[position] != 0xFF:
            return None

        while (
            position < len(image_bytes)
            and image_bytes[position] == 0xFF
        ):
            position += 1

        if position >= len(image_bytes):
            return None

        marker = image_bytes[position]
        position += 1

        if marker in standalone_markers:
            continue

        if marker == 0xDA or position + 2 > len(image_bytes):
            return None

        segment_length = int.from_bytes(
            image_bytes[position:position + 2],
            "big"
        )

        if (
            segment_length < 2
            or position + segment_length > len(image_bytes)
        ):
            return None

        if marker in frame_markers:
            if segment_length < 7:
                return None

            height = int.from_bytes(
                image_bytes[position + 3:position + 5],
                "big"
            )
            width = int.from_bytes(
                image_bytes[position + 5:position + 7],
                "big"
            )

            return width, height

        position += segment_length

    return None


# ============================================================
# ALERT HISTORY
# ============================================================

def load_alert_history():

    if not os.path.exists(ALERT_HISTORY_PATH):
        return []

    try:

        with open(
            ALERT_HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

    except Exception as error:

        print(
            "Alert history load error:",
            error
        )

    return []


# ============================================================
# GET PERSISTENT ALERT COUNT
# ============================================================

def get_persistent_alert_count():

    alerts = load_alert_history()

    return len(alerts)


# ============================================================
# GET LATEST ALERT
# ============================================================

def get_latest_alert():

    alerts = load_alert_history()

    if not alerts:
        return None

    return alerts[-1]


# ============================================================
# SYSTEM STATE
# ============================================================

system_state = {

    "running": False,

    "status": "OFFLINE",

    "detection": "SYSTEM READY",

    "alert_count": get_persistent_alert_count(),

    "last_alert": None,

    "telegram": "Ready"
}


# ============================================================
# INITIAL LAST ALERT
# ============================================================

latest_alert = get_latest_alert()

if latest_alert:

    timestamp = latest_alert.get(
        "timestamp"
    )

    if timestamp:

        system_state["last_alert"] = timestamp


# ============================================================
# ALERT HISTORY API
# ============================================================

@app.route("/api/alerts")
def get_alerts():

    alerts = load_alert_history()

    alerts = list(
        reversed(alerts)
    )

    return jsonify({

        "success": True,

        "alerts": alerts,

        "count": len(alerts)

    })


# ============================================================
# START MONITORING
# ============================================================

@app.route(
    "/api/start",
    methods=["POST"]
)
def start_monitoring():

    global current_session_id
    global last_legacy_frame_saved_at

    with process_lock:

        # ----------------------------------------------------
        # Create a NEW session
        # ----------------------------------------------------

        new_session_id = str(
            uuid.uuid4()
        )

        with session_lock:

            current_session_id = new_session_id

        # ----------------------------------------------------
        # Reset inference
        # ----------------------------------------------------

        reset_inference()
        last_legacy_frame_saved_at = 0.0

        # ----------------------------------------------------
        # Remove old live frame
        # ----------------------------------------------------

        try:

            if os.path.exists(
                LIVE_FRAME_PATH
            ):

                os.remove(
                    LIVE_FRAME_PATH
                )

        except Exception as error:

            print(
                "Could not remove old live frame:",
                error
            )

        # ----------------------------------------------------
        # Refresh alert count
        # ----------------------------------------------------

        system_state["alert_count"] = (
            get_persistent_alert_count()
        )

        # ----------------------------------------------------
        # Start state
        # ----------------------------------------------------

        system_state["running"] = True

        system_state["status"] = "MONITORING"

        system_state["detection"] = "MONITORING"

        system_state["telegram"] = (
            "Ready"
            if alert_agent.telegram_configured
            else "Not configured"
        )

    print(
        "Browser monitoring started."
    )

    print(
        "Session:",
        new_session_id
    )

    return jsonify({

        "success": True,

        "message":
            "Browser monitoring started.",

        "session_id":
            new_session_id

    })


# ============================================================
# RECEIVE CAMERA FRAME
# ============================================================

@app.route(
    "/api/frame",
    methods=["POST"]
)
def receive_frame():
    global last_legacy_frame_saved_at

    # --------------------------------------------------------
    # CHECK SYSTEM RUNNING
    # --------------------------------------------------------

    if not system_state["running"]:

        return jsonify({

            "success": False,

            "message":
                "Monitoring is not running."

        }), 409


    # --------------------------------------------------------
    # CHECK SESSION
    # --------------------------------------------------------

    request_session_id = (
        request.form.get("session_id")
    )

    with session_lock:

        active_session_id = current_session_id


    if (
        not request_session_id
        or request_session_id != active_session_id
    ):

        return jsonify({

            "success": False,

            "message":
                "Expired monitoring session."

        }), 409


    try:

        # ====================================================
        # CHECK FRAME
        # ====================================================

        if "frame" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "No frame received."

            }), 400


        file = request.files["frame"]

        if file.mimetype != "image/jpeg":
            return jsonify({
                "success": False,
                "message": "Frame must be uploaded as a JPEG image."
            }), 415

        image_bytes = file.read()


        if not image_bytes:

            return jsonify({

                "success": False,

                "message":
                    "Empty frame."

            }), 400

        dimensions = get_jpeg_dimensions(image_bytes)

        if dimensions is None:
            return jsonify({
                "success": False,
                "message": "Invalid or unsupported JPEG frame."
            }), 400

        frame_width, frame_height = dimensions

        if (
            frame_width > MAX_FRAME_WIDTH
            or frame_height > MAX_FRAME_HEIGHT
            or frame_width < 1
            or frame_height < 1
        ):
            return jsonify({
                "success": False,
                "message": (
                    "Frame dimensions must be between 1x1 and "
                    f"{MAX_FRAME_WIDTH}x{MAX_FRAME_HEIGHT}."
                )
            }), 413


        # ====================================================
        # DECODE JPEG
        # ====================================================

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Could not decode frame."

            }), 400

        decoded_height, decoded_width = frame.shape[:2]

        if (decoded_width, decoded_height) != dimensions:
            return jsonify({
                    "success": False,
                    "message": "JPEG dimensions could not be verified."
            }), 400


        # ====================================================
        # PROCESS WITH MEDIAPIPE + ML
        # ====================================================

        with process_lock:

            # Re-check session immediately before inference

            if not system_state["running"]:

                return jsonify({

                    "success": False,

                    "message":
                        "Monitoring stopped."

                }), 409


            with session_lock:

                active_session_id = current_session_id


            if request_session_id != active_session_id:

                return jsonify({

                    "success": False,

                    "message":
                        "Expired monitoring session."

                }), 409


            include_legacy_frame = (
                time.monotonic()
                - last_legacy_frame_saved_at
                >= LEGACY_FRAME_INTERVAL_SECONDS
            )

            result = process_frame(
                frame,
                include_frame=include_legacy_frame
            )

            if result.get("frame") is not None:
                last_legacy_frame_saved_at = time.monotonic()


        # ====================================================
        # SAVE PROCESSED FRAME
        # ====================================================

        try:

            if result.get("frame") is not None:

                cv2.imwrite(
                    str(LIVE_FRAME_PATH),
                    result["frame"]
                )

        except Exception as error:

            print(
                "Live frame save error:",
                error
            )


        # ====================================================
        # REMOVE FRAME FROM JSON
        # ====================================================

        result.pop(
            "frame",
            None
        )


        # ====================================================
        # UPDATE SYSTEM STATE
        # ====================================================

        system_state["running"] = True


        if result.get(
            "detection"
        ) == "FALL DETECTED":

            system_state["status"] = "ALERT"

            system_state["detection"] = (
                "FALL DETECTED"
            )

            system_state["alert_count"] = (
                get_persistent_alert_count()
            )


            latest = get_latest_alert()


            if latest:

                system_state["last_alert"] = (
                    latest.get(
                        "timestamp"
                    )
                )


            if result.get(
                "alert_triggered"
            ):

                system_state["telegram"] = "Sending"


        else:

            system_state["status"] = (
                "MONITORING"
            )

            system_state["detection"] = (
                "MONITORING"
            )


        return jsonify(result)


    except Exception as error:

        print(
            "Frame processing error:",
            error
        )

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# ============================================================
# STOP MONITORING
# ============================================================

@app.route(
    "/api/stop",
    methods=["POST"]
)
def stop_monitoring():

    global current_session_id

    with process_lock:

        # ----------------------------------------------------
        # INVALIDATE OLD SESSION FIRST
        # ----------------------------------------------------

        with session_lock:

            current_session_id = None


        # ----------------------------------------------------
        # Reset inference
        # ----------------------------------------------------

        reset_inference()


        # ----------------------------------------------------
        # Reset system
        # ----------------------------------------------------

        system_state["running"] = False

        system_state["status"] = "OFFLINE"

        system_state["detection"] = (
            "SYSTEM READY"
        )

        system_state["telegram"] = "Ready"


        # ----------------------------------------------------
        # Keep persistent alert count
        # ----------------------------------------------------

        system_state["alert_count"] = (
            get_persistent_alert_count()
        )


    print(
        "Browser monitoring stopped."
    )


    return jsonify({

        "success": True,

        "message":
            "Monitoring stopped."

    })


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.route("/api/status")
def get_status():

    system_state["alert_count"] = (
        get_persistent_alert_count()
    )


    latest = get_latest_alert()


    if latest:

        system_state["last_alert"] = (
            latest.get(
                "timestamp"
            )
        )

        latest_telegram_status = latest.get(
            "telegram_status"
        )

        if latest_telegram_status == "PENDING":
            system_state["telegram"] = "Sending"
        elif latest_telegram_status == "SENT":
            system_state["telegram"] = "Sent"
        elif latest_telegram_status == "FAILED":
            system_state["telegram"] = "Failed"
    elif not alert_agent.telegram_configured:
        system_state["telegram"] = "Not configured"


    return jsonify(
        system_state
    )


# ============================================================
# OLD VIDEO FEED
# ============================================================

def generate_camera_frames():

    while True:

        if not system_state["running"]:

            time.sleep(0.1)

            continue


        if not os.path.exists(
            LIVE_FRAME_PATH
        ):

            time.sleep(0.05)

            continue


        try:

            with open(
                LIVE_FRAME_PATH,
                "rb"
            ) as image_file:

                frame_bytes = (
                    image_file.read()
                )


            if not frame_bytes:

                time.sleep(0.05)

                continue


            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )


        except Exception as error:

            print(
                "Camera stream error:",
                error
            )

            time.sleep(0.05)


        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_camera_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return send_from_directory(
        DASHBOARD_DIR,
        "index.html"
    )


# ============================================================
# DASHBOARD STATIC FILES
# ============================================================

@app.route("/<path:path>")
def serve_dashboard_file(path):

    return send_from_directory(
        DASHBOARD_DIR,
        path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print()

    print("=" * 60)

    print(
        "       FallDetection.AI Dashboard"
    )

    print("=" * 60)

    print()

    print("Dashboard:")

    print(
        f"http://127.0.0.1:{port}"
    )

    print()

    print("Persistent Alerts:")

    print(
        get_persistent_alert_count()
    )

    print()

    print("Camera architecture:")

    print("Browser Webcam")

    print("      v")

    print("JavaScript Frame Capture")

    print("      v")

    print("POST /api/frame")

    print("      v")

    print("MediaPipe Pose + Skeleton")

    print("      v")

    print("30-Frame ML Model")

    print("      v")

    print("Fall Detection")

    print("      v")

    print("Alert Response Agent")

    print("      v")

    print("Telegram")

    print()

    print("=" * 60)

    print()


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )