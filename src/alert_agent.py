import os
import json
import requests
import threading

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALERT_HISTORY_PATH = OUTPUT_DIR / "alert_history.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AlertResponseAgent:
    """
    Alert Response Agent

    Responsibilities:
    1. Create a unique incident
    2. Record event details
    3. Decide response action
    4. Send Telegram notification
    5. Store incident history
    """

    def __init__(self):
        self.alert_count = 0
        self._history_lock = threading.Lock()

        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        self.alert_history = self.load_alert_history()

        # Continue incident numbering from previous history
        self.alert_count = len(self.alert_history)

    @property
    def telegram_configured(self):
        return bool(
            self.telegram_token
            and self.telegram_chat_id
        )

    # ---------------------------------------------------------
    # LOAD ALERT HISTORY
    # ---------------------------------------------------------

    def load_alert_history(self):
        if not ALERT_HISTORY_PATH.exists():
            return []

        try:
            with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)

                if isinstance(data, list):
                    return data

        except (json.JSONDecodeError, OSError) as error:
            print("Could not load alert history:", error)

        return []

    # ---------------------------------------------------------
    # SAVE ALERT HISTORY
    # ---------------------------------------------------------

    def save_alert_history(self):
        with self._history_lock:
            self._save_alert_history_unlocked()

    def _save_alert_history_unlocked(self):
        temporary_path = ALERT_HISTORY_PATH.with_suffix(
            ".json.tmp"
        )

        try:
            with open(
                temporary_path,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    self.alert_history,
                    file,
                    indent=4
                )
            os.replace(
                temporary_path,
                ALERT_HISTORY_PATH
            )

        except OSError as error:
            print("Could not save alert history:", error)

    # ---------------------------------------------------------
    # CREATE ALERT
    # ---------------------------------------------------------

    def create_alert(
        self,
        camera_name="Laptop Camera",
        confidence=None
    ):
        with self._history_lock:
            self.alert_count += 1
            incident_id = f"FD-{self.alert_count:03d}"

        current_time = datetime.now()
        formatted_time = current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ---------------------------------------------
        # RESPONSE DECISION
        # ---------------------------------------------

        if confidence is not None:
            if confidence >= 0.80:
                severity = "HIGH"
            elif confidence >= 0.60:
                severity = "MEDIUM"
            else:
                severity = "LOW"
        else:
            severity = "HIGH"

        response_action = "SEND TELEGRAM ALERT"

        # ---------------------------------------------
        # CREATE INCIDENT
        # ---------------------------------------------

        incident = {
            "incident_id": incident_id,
            "event": "FALL DETECTED",
            "camera": camera_name,
            "timestamp": formatted_time,
            "severity": severity,
            "confidence": (
                round(confidence * 100, 2)
                if confidence is not None
                else None
            ),
            "status": "CONFIRMED",
            "response_action": response_action,
            "telegram_status": "PENDING"
        }

        print("")
        print("=" * 50)
        print("ALERT RESPONSE AGENT ACTIVATED")
        print("=" * 50)

        print("Incident ID:", incident_id)
        print("Event: FALL DETECTED")
        print("Camera:", camera_name)
        print("Time:", formatted_time)
        print("Severity:", severity)

        if confidence is not None:
            print(
                f"Confidence: {confidence * 100:.1f}%"
            )

        print("Response:", response_action)

        # ---------------------------------------------
        # BUILD TELEGRAM MESSAGE
        # ---------------------------------------------

        confidence_text = ""

        if confidence is not None:
            confidence_text = (
                f"Confidence: {confidence * 100:.1f}%\n"
            )

        alert_message = (
            "FALL DETECTED\n\n"
            f"Incident ID: {incident_id}\n"
            f"Camera: {camera_name}\n"
            f"Time: {formatted_time}\n"
            f"Severity: {severity}\n"
            f"{confidence_text}"
            "Status: Confirmed\n\n"
            "Please check the person immediately."
        )

        with self._history_lock:
            self.alert_history.append(incident)
            self._save_alert_history_unlocked()

        try:
            telegram_sent = self.send_telegram_alert(
                alert_message
            )
        except Exception as error:
            print(
                "[ALERT] Unexpected Telegram error:",
                type(error).__name__
            )
            telegram_sent = False

        with self._history_lock:
            incident["telegram_status"] = (
                "SENT"
                if telegram_sent
                else "FAILED"
            )
            self._save_alert_history_unlocked()

        print("Incident logged:", incident_id)
        print("=" * 50)

        return alert_message

    # ---------------------------------------------------------
    # TELEGRAM
    # ---------------------------------------------------------

    def send_telegram_alert(self, message):

        if not self.telegram_configured:
            print(
                "[ALERT] Telegram credentials not configured."
            )
            return False

        url = (
            "https://api.telegram.org/bot"
            f"{self.telegram_token}/sendMessage"
        )

        try:
            response = requests.post(
                url,
                data={
                    "chat_id": self.telegram_chat_id,
                    "text": message
                },
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            if (
                response.status_code == 200
                and isinstance(result, dict)
                and result.get("ok") is True
            ):
                print("TELEGRAM ALERT SENT")
                return True

            description = (
                result.get("description", "API rejected the alert.")
                if isinstance(result, dict)
                else "API returned an invalid response."
            )
            print(
                "[ALERT] Telegram API rejected alert:",
                description
            )

            return False

        except requests.RequestException as error:
            print(
                "[ALERT] Telegram request failed:",
                type(error).__name__
            )

            return False
        except ValueError as error:
            print(
                "[ALERT] Telegram returned invalid JSON:",
                error
            )
            return False

    # ---------------------------------------------------------
    # GET ALERT HISTORY
    # ---------------------------------------------------------

    def get_alert_history(self):
        with self._history_lock:
            return [
                dict(alert)
                for alert in self.alert_history
            ]


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    agent = AlertResponseAgent()

    message = agent.create_alert()

    print("")
    print("=" * 40)
    print(message)
    print("=" * 40)