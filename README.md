# FallDetection.AI

FallDetection.AI is a real-time, computer-vision-based fall detection and alert system. It processes camera frames with MediaPipe Pose, classifies temporal landmark sequences with a trained Random Forest model, and presents monitoring state and incidents in a Flask dashboard.

## Core Features

- Live fall detection from a browser webcam
- MediaPipe Pose extraction of 33 body landmarks
- A 30-frame temporal buffer for Random Forest classification
- Existing motion-based fall checks and recovery latch
- Flask dashboard with live frame, status, and alert history
- Alert Response Agent with severity and confidence reporting
- Optional Telegram notifications configured through environment variables
- Persistent local alert history in `outputs/alert_history.json`
- Unique MP4 for each monitoring session and each fall incident

## System Architecture

```text
Browser Webcam
	↓
JPEG frames to Flask
	↓
MediaPipe Pose
	↓
33 Pose Landmarks
	↓
30-Frame Temporal Buffer
	↓
Random Forest + existing motion checks
	↓
Fall Detection
	↓
Alert Response Agent
	├── Telegram
	└── Flask Dashboard + browser skeleton overlay
```

## Project Structure

```text
FallDetection_Ai_System/
|-- app.py
|-- render.yaml
|-- requirements.txt
|-- .python-version
|-- .env.example
|-- .gitignore
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- alert_agent.py
|   |-- browser_inference.py
|   |-- evaluate_model.py
|   |-- extract_features.py
|   |-- realtime_inference.py
|   |-- train_from_features.py
|   `-- vision_core.py
|-- models/
|   `-- fall_detection_model.pkl
|-- DASHBOARD/
|   |-- index.html
|   |-- script.js
|   `-- style.css
|-- outputs/
|   |-- .gitkeep
|   `-- README.md
`-- docs/
    |-- PRD.md
    `-- deployment.md
```

The uppercase `DASHBOARD/` directory is retained from the existing project and is referenced with its exact case for case-sensitive hosts. The browser captures webcam frames and sends resized JPEGs to Flask; `src/browser_inference.py` runs MediaPipe and the existing model, then returns normalized landmarks for the browser skeleton overlay. The Flask application does not use `cv2.VideoCapture(0)`. Local virtual environments, caches, the optional training data in `data/`, and generated runtime files are excluded from Git. The trained model stays in the repository because inference loads it at runtime. `render.yaml` describes a single-worker Render deployment and requires access and Telegram secrets to be entered in the hosting dashboard.

## Technology Stack

- Python 3.12
- Flask and Gunicorn (WSGI deployment)
- OpenCV and MediaPipe Pose
- NumPy, scikit-learn, and joblib
- pandas for feature training/evaluation
- Matplotlib and Seaborn for evaluation output
- Requests and python-dotenv for optional Telegram configuration

## Installation

Use Python 3.12.13, as verified with the current model and MediaPipe environment. From PowerShell:

```powershell
git clone <your-repository-url>
cd FallDetection_Ai_System
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If `.env` already exists, do not overwrite it. Edit it locally to configure credentials; never commit it.

## Environment Variables

`src/alert_agent.py` reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the process environment or `.env`. Notifications are unavailable until both are configured; detection and local alert-history recording continue. `.env.example` contains placeholders only.

For optional feature extraction, set `UR_DATASET_DIR` to the directory containing `adl_files/` and `fall_files/`. Its default is `data/ur_dataset`. The source UR dataset is not bundled. The extracted feature CSV is used by training/evaluation, not live inference.

## Running Locally

From the project root with the virtual environment active:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>, allow browser camera access, then select **Start Monitoring**. The browser sends frames to the Flask app running on the same machine. Select **Stop Monitoring** to end the camera session. The dashboard uses a normalized-landmark canvas overlay; it does not stream processed video back from Flask.

## Dashboard

The browser dashboard is served by Flask at `/`. Its JavaScript requests the visitor's webcam with `getUserMedia()`, sends at most one resized JPEG frame request at a time, and draws returned normalized landmarks on a canvas over the live video. Same-origin relative API URLs are used. Browser camera access requires localhost or HTTPS.

## Testing

1. Start the application and open the dashboard.
2. Start monitoring and confirm the camera feed and monitoring status.
3. Use a safe prerecorded clip or a controlled, non-hazardous demonstration; do not attempt an actual fall.
4. Confirm the detection status and incident entry in the dashboard.
5. If Telegram credentials are configured, verify receipt of the notification.
6. Confirm the incident is recorded in `outputs/alert_history.json`.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Serve the dashboard |
| POST | `/api/start` | Start a browser-camera monitoring session |
| POST | `/api/frame` | Process one JPEG frame from the active browser session |
| POST | `/api/stop` | Stop the browser-camera session |
| GET | `/api/status` | Return monitoring and latest-alert status |
| GET | `/api/alerts` | Return persistent alert history |
| GET | `/video_feed` | Legacy, low-rate processed-frame stream |

## Model

Inference loads `models/fall_detection_model.pkl` with joblib. It is a trusted local pickle artifact; do not load a model from an untrusted source. Training and evaluation scripts use `data/urfd_extracted_features.csv` and may regenerate model/evaluation artifacts.

## Security

- Keep Telegram credentials in `.env` locally or deployment environment settings.
- Keep `APP_ACCESS_PASSWORD` private and unique; hosted deployments require it before startup.
- `.env` is Git-ignored; never commit credentials or paste them into source files.
- Rotate/revoke credentials if they have been exposed or committed.
- Use HTTPS and a production WSGI server for public hosting.
- The frame API rejects cross-origin writes and frames larger than 640x480 or 1 MiB.
- Treat confidence as the model's reported score, not a medically calibrated probability.

## Deployment Readiness

The Flask application is importable by a WSGI server, uses the deployment-provided `PORT`, and serves the dashboard using portable project-relative paths. This working folder is not currently a Git repository. To deploy, create a private GitHub repository, add and push this project, then create a Render Blueprint from `render.yaml`, and supply a unique `APP_ACCESS_PASSWORD` plus Telegram credentials in Render's secret environment fields. Render serves the camera page over HTTPS; users must enter the configured HTTP Basic credentials before using the dashboard. The deployment manifest keeps one worker because the pose tracker and frame buffer are process-global. See the deployment guide before hosting.

## Limitations

- Fall detection can produce false positives and false negatives.
- Model confidence is not necessarily calibrated.
- Monitoring requires the user's browser to grant webcam access and to reach the Flask frame API.
- Remote use requires HTTPS and enough backend capacity to process incoming camera frames.
- This system is not a medical device and does not provide a diagnosis.

## Future Improvements

- Persistent cloud alert storage
- Better model calibration
- Additional datasets
- Production monitoring
