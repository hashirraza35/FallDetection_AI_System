# Runtime Outputs

- `fall_detection_session_*.mp4` in this directory is the single full monitoring-session recording. Fall detection still creates alerts, but it does not create separate fall-event video clips.
- `live_frame.jpg` and `alert_history.json` support the live dashboard and alert history.

The MP4 is written directly to this directory. Stop monitoring from the dashboard and allow it to finish so the MP4 is finalized and playable. Do not forcibly terminate the process while it is recording.

These generated and potentially user-specific files are excluded from Git; `.gitkeep` preserves the directory in a clean checkout.

Hosted filesystems may be ephemeral. Alert history and generated media are not durable across service restarts or redeployments on hosts with ephemeral storage.
