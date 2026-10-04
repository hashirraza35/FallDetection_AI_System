# Deployment Preparation

## Current Local Deployment

Use Python 3.12.13 from the project root. In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Do not overwrite an existing `.env`. Set Telegram values locally if notifications are required. Open <http://127.0.0.1:5000> and start monitoring from the dashboard. Stop monitoring before stopping Flask with `Ctrl+C`.

## Render Preparation

No deployment has been performed. The repository includes a Render Blueprint at `render.yaml`. Push the project to a GitHub repository, then create a new Blueprint in Render and select that repository. During setup, enter a strong, unique `APP_ACCESS_PASSWORD` and the Telegram values if notifications are wanted. Do not put secret values in `render.yaml` or source control.

The Blueprint defines a Python web service with:

**Build command**

```sh
pip install -r requirements.txt
```

**Start command**

```sh
gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 8 --timeout 120
```

The single worker is required because MediaPipe, the model, and the frame buffer are process-global. `APP_REQUIRE_AUTH=true` is enabled by the Blueprint; the service refuses to start unless `APP_ACCESS_PASSWORD` is supplied. The dashboard uses HTTP Basic authentication; a browser prompts for the configured `APP_ACCESS_USERNAME` (default `admin`) and password. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as secret environment variables to enable Telegram notifications. `UR_DATASET_DIR` is only needed by the optional feature-extraction script. Python 3.12.13 is recorded in `.python-version`.

The `/healthz` endpoint is used for service health checks and intentionally remains public. The app also sets browser security headers, rejects cross-origin writes, limits uploads to 1 MiB and decoded frames to 640x480. The free service is intended for initial testing; sustained inference may need a larger instance. No cloud deployment or capacity test has been performed. Use persistent storage if alert history must survive instance replacement.

## Important Camera Limitation

The dashboard requests the user's webcam in the browser and posts resized JPEG frames to Flask at `/api/frame`. The server runs MediaPipe and the existing model on those frames and returns normalized landmarks for the browser's skeleton canvas. It does not open a server-side webcam. Serve the dashboard over HTTPS for browser camera access and ensure the browser can reach the Flask API. The dashboard currently uses same-origin relative API URLs; separate frontend/backend domains are not supported by this deployment without additional API URL and CORS configuration.

This browser-camera flow is distinct from `src/realtime_inference.py`, which retains its standalone local-camera prototype. The Flask dashboard uses `src/browser_inference.py`.

## Storage and Secrets

`outputs/alert_history.json`, live frames, and demo videos are local runtime files and are Git-ignored. Render's filesystem is ephemeral by default, so alert history is not durable across redeploys or instance replacement. Persistent cloud storage is a future improvement, not part of this deployment preparation.

Keep `.env` out of Git. Use Render's environment settings for credentials, enable HTTPS, and rotate/revoke any Telegram credential that has been exposed. Never load an untrusted pickle model.
