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
    reset_inference,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DASHBOARD_DIR = PROJECT_ROOT / "DASHBOARD"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LIVE_FRAME_PATH = OUTPUTS_DIR / "live_frame.jpg"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"


# ============================================================
# CONFIGURATION
# ============================================================

LEGACY_FRAME_INTERVAL_SECONDS = 1.0

MAX_UPLOAD_BYTES = 1_048_576

MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480

last_legacy_frame_saved_at = 0.0

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder=DASHBOARD_DIR,
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ============================================================
# OPTIONAL APP AUTH
# ============================================================

REQUIRE_APP_AUTH = (
    os.getenv("APP_REQUIRE_AUTH", "false")
    .strip()
    .lower()
    in {"1", "true", "yes"}
)

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
        "APP_ACCESS_PASSWORD must be configured "
        "when APP_REQUIRE_AUTH is enabled."
    )


# ============================================================
# GLOBAL LOCKS / SESSION
# ============================================================

process_lock = threading.Lock()
session_lock = threading.Lock()

current_session_id = None


# ============================================================
# SECURITY / AUTH
# ============================================================

def get_request_origin():
    origin = request.headers.get("Origin")

    if not origin:
        return None

    try:
        parsed_origin = urlsplit(origin)

        if not parsed_origin.scheme or not parsed_origin.netloc:
            return None

        return parsed_origin

    except Exception:
        return None


def check_origin():
    """
    Same-origin protection.

    This is intentionally kept simple for the current
    Render-hosted dashboard.

    Later, when frontend is moved to Vercel, this can be
    expanded to allow the Vercel domain.
    """

    origin = get_request_origin()

    if origin is None:
        return True

    request_host = request.host

    forwarded_proto = request.headers.get(
        "X-Forwarded-Proto",
        request.scheme
    )

    if origin.netloc != request_host:
        return False

    if origin.scheme != forwarded_proto:
        return False

    return True


def check_basic_auth():
    """
    Optional HTTP Basic authentication.

    Disabled by default.
    """

    if not REQUIRE_APP_AUTH:
        return True

    auth = request.authorization

    if not auth:
        return False

    username_ok = hmac.compare_digest(
        auth.username or "",
        APP_ACCESS_USERNAME
    )

    password_ok = hmac.compare_digest(
        auth.password or "",
        APP_ACCESS_PASSWORD
    )

    return username_ok and password_ok


@app.before_request
def security_check():

    # OPTIONS requests are allowed for future CORS/preflight use.
    if request.method == "OPTIONS":
        return None

    # GET/HEAD are allowed without auth.
    if request.method in {"GET", "HEAD"}:
        return None

    # Origin protection.
    if not check_origin():
        return jsonify({
            "success": False,
            "message": "Origin not allowed."
        }), 403

    # Optional authentication.
    if not check_basic_auth():
        response = jsonify({
            "success": False,
            "message": "Authentication required."
        })

        response.status_code = 401
        response.headers["WWW-Authenticate"] = (
            'Basic realm="FallDetection.AI"'
        )

        return response

    return None


# ============================================================
# SECURITY HEADERS
# ============================================================

@app.after_request
def add_security_headers(response):

    response.headers["X-Content-Type-Options"] = "nosniff"

    response.headers["X-Frame-Options"] = "SAMEORIGIN"

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    response.headers["Permissions-Policy"] = (
        "camera=(self), microphone=()"
    )

    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )

    return response


# ============================================================
# REQUEST SIZE ERROR
# ============================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):

    return jsonify({
        "success": False,
        "message": "Uploaded frame is too large."
    }), 413


# ============================================================
# 404 API HANDLER
# ============================================================

@app.errorhandler(404)
def handle_not_found(error):

    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "message": "API endpoint not found."
        }), 404

    return error


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/healthz", methods=["GET"])
def healthz():

    return jsonify({
        "status": "ok"
    })


# ============================================================
# ALERT HISTORY
# ============================================================

def load_alert_history():

    if not ALERT_HISTORY_PATH.exists():
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

        return []

    except Exception as error:

        print(
            "Alert history load error:",
            error,
            flush=True
        )

        return []


def get_persistent_alert_count():

    history = load_alert_history()

    return len(history)


def get_latest_alert():

    history = load_alert_history()

    if not history:
        return None

    return history[-1]


# ============================================================
# SYSTEM STATE
# ============================================================

system_state = {

    "running": False,

    "status": "OFFLINE",

    "detection": "SYSTEM READY",

    "alert_count": get_persistent_alert_count(),

    "last_alert": None,

    "telegram": "Ready",
}


# ============================================================
# ALERT API
# ============================================================

@app.route("/api/alerts", methods=["GET"])
def get_alerts():

    history = load_alert_history()

    return jsonify({
        "success": True,
        "alerts": list(reversed(history)),
        "count": len(history),
    })


# ============================================================
# STATUS API
# ============================================================

@app.route("/api/status", methods=["GET"])
def get_status():

    system_state["alert_count"] = (
        get_persistent_alert_count()
    )

    latest = get_latest_alert()

    if latest:

        system_state["last_alert"] = (
            latest.get("timestamp")
        )

    return jsonify(system_state)


# ============================================================
# START MONITORING
# ============================================================

@app.route("/api/start", methods=["POST"])
def start_monitoring():

    global current_session_id
    global last_legacy_frame_saved_at

    with process_lock:

        new_session_id = str(
            uuid.uuid4()
        )

        with session_lock:

            current_session_id = new_session_id

        reset_inference()

        last_legacy_frame_saved_at = 0.0

        try:

            if LIVE_FRAME_PATH.exists():

                LIVE_FRAME_PATH.unlink()

        except Exception as error:

            print(
                "Could not remove old live frame:",
                error,
                flush=True
            )

        system_state["alert_count"] = (
            get_persistent_alert_count()
        )

        system_state["running"] = True

        system_state["status"] = "MONITORING"

        system_state["detection"] = "MONITORING"

        system_state["telegram"] = (
            "Ready"
            if alert_agent.telegram_configured
            else "Not configured"
        )

    print(
        "========================================",
        flush=True
    )

    print(
        "Browser monitoring started.",
        flush=True
    )

    print(
        "Session:",
        new_session_id,
        flush=True
    )

    print(
        "Running state:",
        system_state["running"],
        flush=True
    )

    print(
        "========================================",
        flush=True
    )

    return jsonify({

        "success": True,

        "message": (
            "Browser monitoring started."
        ),

        "session_id": new_session_id,

    })


# ============================================================
# JPEG DIMENSION READER
# ============================================================

def get_jpeg_dimensions(image_bytes):

    """
    Read JPEG width/height without decoding the entire image.
    """

    try:

        if len(image_bytes) < 4:
            return None

        if image_bytes[0:2] != b"\xff\xd8":
            return None

        index = 2

        while index < len(image_bytes):

            while (
                index < len(image_bytes)
                and image_bytes[index] == 0xFF
            ):
                index += 1

            if index >= len(image_bytes):
                return None

            marker = image_bytes[index]

            index += 1

            # Standalone JPEG markers.
            if marker in {
                0x01,
                *range(0xD0, 0xD9),
            }:
                continue

            if index + 1 >= len(image_bytes):
                return None

            segment_length = int.from_bytes(
                image_bytes[index:index + 2],
                "big"
            )

            if segment_length < 2:
                return None

            if index + segment_length > len(image_bytes):
                return None

            # SOF markers.
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:

                if segment_length < 7:
                    return None

                height = int.from_bytes(
                    image_bytes[index + 3:index + 5],
                    "big"
                )

                width = int.from_bytes(
                    image_bytes[index + 5:index + 7],
                    "big"
                )

                return width, height

            index += segment_length

        return None

    except Exception as error:

        print(
            "JPEG dimension error:",
            error,
            flush=True
        )

        return None


# ============================================================
# FRAME API
# ============================================================

@app.route("/api/frame", methods=["POST"])
def receive_frame():

    global last_legacy_frame_saved_at

    # ========================================================
    # DEBUG 1 — VERY IMPORTANT
    # ========================================================

    print(
        "----------------------------------------",
        flush=True
    )

    print(
        "FRAME DEBUG | running=",
        system_state["running"],
        "| request_session=",
        request.form.get("session_id"),
        "| active_session=",
        current_session_id,
        flush=True
    )

    print(
        "FRAME DEBUG | content_type=",
        request.content_type,
        "| content_length=",
        request.content_length,
        flush=True
    )

    # ========================================================
    # CHECK SYSTEM RUNNING
    # ========================================================

    if not system_state["running"]:

        print(
            "FRAME DEBUG RESULT | Monitoring is NOT running.",
            flush=True
        )

        return jsonify({

            "success": False,

            "message": (
                "Monitoring is not running."
            )

        }), 409

    # ========================================================
    # SESSION CHECK
    # ========================================================

    request_session_id = request.form.get(
        "session_id"
    )

    with session_lock:

        active_session_id = current_session_id

    print(
        "SESSION DEBUG | request=",
        request_session_id,
        "| active=",
        active_session_id,
        "| match=",
        request_session_id == active_session_id,
        flush=True
    )

    if (
        not request_session_id
        or request_session_id != active_session_id
    ):

        print(
            "FRAME DEBUG RESULT | EXPIRED SESSION.",
            flush=True
        )

        return jsonify({

            "success": False,

            "message": (
                "Expired monitoring session."
            )

        }), 409

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    try:

        # ----------------------------------------------------
        # FRAME FIELD
        # ----------------------------------------------------

        if "frame" not in request.files:

            print(
                "FRAME DEBUG RESULT | No frame received.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": "No frame received."

            }), 400

        file = request.files["frame"]

        print(
            "FRAME DEBUG | filename=",
            file.filename,
            "| mimetype=",
            file.mimetype,
            flush=True
        )

        # ----------------------------------------------------
        # JPEG MIME TYPE
        # ----------------------------------------------------

        if file.mimetype != "image/jpeg":

            print(
                "FRAME DEBUG RESULT | Invalid MIME type.",
                file.mimetype,
                flush=True
            )

            return jsonify({

                "success": False,

                "message": (
                    "Frame must be uploaded "
                    "as a JPEG image."
                )

            }), 415

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            print(
                "FRAME DEBUG RESULT | Empty frame.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": "Empty frame."

            }), 400

        print(
            "FRAME DEBUG | image_bytes=",
            len(image_bytes),
            flush=True
        )

        # ----------------------------------------------------
        # JPEG DIMENSIONS
        # ----------------------------------------------------

        dimensions = get_jpeg_dimensions(
            image_bytes
        )

        if dimensions is None:

            print(
                "FRAME DEBUG RESULT | Invalid JPEG.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": (
                    "Invalid or unsupported JPEG frame."
                )

            }), 400

        frame_width, frame_height = dimensions

        print(
            "FRAME DEBUG | JPEG dimensions=",
            frame_width,
            "x",
            frame_height,
            flush=True
        )

        # ----------------------------------------------------
        # DIMENSION LIMIT
        # ----------------------------------------------------

        if (
            frame_width > MAX_FRAME_WIDTH
            or frame_height > MAX_FRAME_HEIGHT
            or frame_width < 1
            or frame_height < 1
        ):

            print(
                "FRAME DEBUG RESULT | Frame dimensions rejected.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": (
                    "Frame dimensions must be "
                    f"between 1x1 and "
                    f"{MAX_FRAME_WIDTH}x"
                    f"{MAX_FRAME_HEIGHT}."
                )

            }), 413

        # ----------------------------------------------------
        # DECODE WITH OPENCV
        # ----------------------------------------------------

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            print(
                "FRAME DEBUG RESULT | OpenCV decode failed.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": (
                    "Could not decode frame."
                )

            }), 400

        decoded_height, decoded_width = (
            frame.shape[:2]
        )

        print(
            "FRAME DEBUG | decoded dimensions=",
            decoded_width,
            "x",
            decoded_height,
            flush=True
        )

        # ----------------------------------------------------
        # VERIFY DIMENSIONS
        # ----------------------------------------------------

        if (
            decoded_width,
            decoded_height
        ) != dimensions:

            print(
                "FRAME DEBUG RESULT | JPEG dimensions mismatch.",
                flush=True
            )

            return jsonify({

                "success": False,

                "message": (
                    "JPEG dimensions could not "
                    "be verified."
                )

            }), 400

        # ====================================================
        # LOCKED INFERENCE SECTION
        # ====================================================

        with process_lock:

            # -----------------------------------------------
            # RUNNING CHECK AGAIN
            # -----------------------------------------------

            if not system_state["running"]:

                print(
                    "FRAME DEBUG RESULT | "
                    "Monitoring stopped during processing.",
                    flush=True
                )

                return jsonify({

                    "success": False,

                    "message": (
                        "Monitoring stopped."
                    )

                }), 409

            # -----------------------------------------------
            # SESSION CHECK AGAIN
            # -----------------------------------------------

            with session_lock:

                active_session_id = (
                    current_session_id
                )

            print(
                "LOCKED SESSION DEBUG | request=",
                request_session_id,
                "| active=",
                active_session_id,
                "| match=",
                request_session_id == active_session_id,
                flush=True
            )

            if (
                request_session_id
                != active_session_id
            ):

                print(
                    "FRAME DEBUG RESULT | "
                    "Session expired inside lock.",
                    flush=True
                )

                return jsonify({

                    "success": False,

                    "message": (
                        "Expired monitoring session."
                    )

                }), 409

            # -----------------------------------------------
            # LEGACY FRAME SAVE DECISION
            # -----------------------------------------------

            include_legacy_frame = (

                time.monotonic()
                - last_legacy_frame_saved_at
                >= LEGACY_FRAME_INTERVAL_SECONDS

            )

            print(
                "INFERENCE DEBUG | "
                "Starting process_frame | "
                "include_frame=",
                include_legacy_frame,
                flush=True
            )

            # -----------------------------------------------
            # AI INFERENCE
            # -----------------------------------------------

            result = process_frame(

                frame,

                include_frame=(
                    include_legacy_frame
                )

            )

            print(
                "INFERENCE DEBUG | "
                "process_frame completed | "
                "detection=",
                result.get("detection"),
                "| alert_triggered=",
                result.get("alert_triggered"),
                flush=True
            )

            # -----------------------------------------------
            # UPDATE LEGACY FRAME TIMER
            # -----------------------------------------------

            if result.get("frame") is not None:

                last_legacy_frame_saved_at = (
                    time.monotonic()
                )

        # ====================================================
        # SAVE LIVE FRAME
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
                error,
                flush=True
            )

        # ====================================================
        # REMOVE BINARY FRAME FROM JSON
        # ====================================================

        result.pop(
            "frame",
            None
        )

        # ====================================================
        # SYSTEM STATE
        # ====================================================

        system_state["running"] = True

        # ====================================================
        # FALL DETECTED
        # ====================================================

        if result.get("detection") == "FALL DETECTED":

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
                    latest.get("timestamp")
                )

            if result.get("alert_triggered"):

                system_state["telegram"] = (
                    "Sending"
                )

        # ====================================================
        # NORMAL MONITORING
        # ====================================================

        else:

            system_state["status"] = (
                "MONITORING"
            )

            system_state["detection"] = (
                "MONITORING"
            )

        print(
            "FRAME SUCCESS | detection=",
            result.get("detection"),
            flush=True
        )

        print(
            "----------------------------------------",
            flush=True
        )

        return jsonify(result)

    # ========================================================
    # ANY UNEXPECTED ERROR
    # ========================================================

    except Exception as error:

        print(
            "========================================",
            flush=True
        )

        print(
            "FRAME PROCESSING ERROR:",
            repr(error),
            flush=True
        )

        print(
            "========================================",
            flush=True
        )

        return jsonify({

            "success": False,

            "message": str(error)

        }), 500


# ============================================================
# STOP MONITORING
# ============================================================

@app.route("/api/stop", methods=["POST"])
def stop_monitoring():

    global current_session_id

    with process_lock:

        with session_lock:

            current_session_id = None

        reset_inference()

        system_state["running"] = False

        system_state["status"] = "OFFLINE"

        system_state["detection"] = (
            "SYSTEM READY"
        )

        system_state["telegram"] = (
            "Ready"
            if alert_agent.telegram_configured
            else "Not configured"
        )

        system_state["alert_count"] = (
            get_persistent_alert_count()
        )

    print(
        "Browser monitoring stopped.",
        flush=True
    )

    return jsonify({

        "success": True,

        "message": (
            "Browser monitoring stopped."
        )

    })


# ============================================================
# LIVE FRAME
# ============================================================

@app.route("/live_frame")
def live_frame():

    if not LIVE_FRAME_PATH.exists():

        return Response(
            status=404
        )

    try:

        with open(
            LIVE_FRAME_PATH,
            "rb"
        ) as file:

            data = file.read()

        return Response(

            data,

            mimetype="image/jpeg",

            headers={
                "Cache-Control": (
                    "no-store, no-cache, "
                    "must-revalidate, max-age=0"
                )
            }

        )

    except Exception as error:

        print(
            "Live frame read error:",
            error,
            flush=True
        )

        return Response(
            status=500
        )


# ============================================================
# LEGACY VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            if LIVE_FRAME_PATH.exists():

                try:

                    with open(
                        LIVE_FRAME_PATH,
                        "rb"
                    ) as file:

                        frame_data = file.read()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_data
                        + b"\r\n"
                    )

                except Exception:
                    pass

            time.sleep(
                LEGACY_FRAME_INTERVAL_SECONDS
            )

    return Response(

        generate(),

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

    index_path = (
        DASHBOARD_DIR / "index.html"
    )

    if not index_path.exists():

        return jsonify({

            "success": False,

            "message": (
                "Dashboard index.html not found."
            )

        }), 404

    return send_from_directory(
        DASHBOARD_DIR,
        "index.html"
    )


# ============================================================
# STATIC DASHBOARD FILES
# ============================================================

@app.route(
    "/<path:path>",
    methods=["GET"]
)
def dashboard_static(path):

    requested_path = (
        DASHBOARD_DIR / path
    )

    if requested_path.exists() and requested_path.is_file():

        return send_from_directory(
            DASHBOARD_DIR,
            path
        )

    return jsonify({

        "success": False,

        "message": "Resource not found."

    }), 404


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    print(
        "========================================"
    )

    print(
        "FallDetection.AI starting..."
    )

    print(
        "Host: 0.0.0.0"
    )

    print(
        "Port:",
        port
    )

    print(
        "Auth:",
        "ENABLED"
        if REQUIRE_APP_AUTH
        else "DISABLED"
    )

    print(
        "========================================"
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )