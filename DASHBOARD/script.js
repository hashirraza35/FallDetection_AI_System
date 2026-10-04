// ============================================================
// FallDetection.AI - Dashboard JavaScript
// Browser Webcam + Skeleton Overlay + Backend AI Detection
// ============================================================


// ============================================================
// DOM ELEMENTS
// ============================================================

const startBtn =
    document.getElementById("startBtn");

const stopBtn =
    document.getElementById("stopBtn");

const systemStatus =
    document.getElementById("systemStatus");

const cameraStatus =
    document.getElementById("cameraStatus");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const cameraFeed =
    document.getElementById("cameraFeed");

const monitoringOverlay =
    document.getElementById("monitoringOverlay");

const detectionStatus =
    document.getElementById("detectionStatus");

const statusIcon =
    document.getElementById("statusIcon");

const detectionText =
    document.getElementById("detectionText");

const detectionDescription =
    document.getElementById("detectionDescription");

const alertCount =
    document.getElementById("alertCount");

const lastAlert =
    document.getElementById("lastAlert");

const telegramStatus =
    document.getElementById("telegramStatus");

const historyCount =
    document.getElementById("historyCount");

const alertHistory =
    document.getElementById("alertHistory");

const currentDateTime =
    document.getElementById("currentDateTime");


// ============================================================
// CAMERA STATE
// ============================================================

let cameraStream = null;

let browserCameraActive = false;

let startInProgress = false;

let stopInProgress = false;


// ============================================================
// BACKEND SESSION
// ============================================================

let monitoringSessionId = null;


// ============================================================
// FRAME SENDING
// ============================================================

const frameCanvas =
    document.createElement("canvas");

const frameContext =
    frameCanvas.getContext("2d");

const MAX_FRAME_WIDTH = 640;
const MAX_FRAME_HEIGHT = 480;
const FRAME_JPEG_QUALITY = 0.60;
const FRAME_INTERVAL_MS = 120;

let frameSending = false;

let frameTimer = null;


// ============================================================
// SKELETON OVERLAY CANVAS
// ============================================================

let skeletonCanvas = null;

let skeletonContext = null;


// ============================================================
// LAST DETECTION RESULT
// ============================================================

let lastDetectionResult = null;


// ============================================================
// MEDIAPIPE POSE CONNECTIONS
// ============================================================

const POSE_CONNECTIONS = [

    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],

    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],

    [9, 10],

    [11, 12],

    [11, 13],
    [13, 15],

    [15, 17],
    [15, 19],
    [15, 21],

    [12, 14],
    [14, 16],

    [16, 18],
    [16, 20],
    [16, 22],

    [11, 23],
    [12, 24],

    [23, 24],

    [23, 25],
    [25, 27],

    [27, 29],
    [27, 31],

    [24, 26],
    [26, 28],

    [28, 30],
    [28, 32]

];


// ============================================================
// CREATE SKELETON CANVAS
// ============================================================

function createSkeletonCanvas() {

    if (!cameraFeed) {
        return;
    }

    if (skeletonCanvas) {
        return;
    }

    skeletonCanvas =
        document.createElement("canvas");

    skeletonCanvas.id =
        "skeletonOverlay";

    skeletonCanvas.style.position =
        "absolute";

    skeletonCanvas.style.left =
        "0";

    skeletonCanvas.style.top =
        "0";

    skeletonCanvas.style.width =
        "100%";

    skeletonCanvas.style.height =
        "100%";

    skeletonCanvas.style.pointerEvents =
        "none";

    skeletonCanvas.style.zIndex =
        "10";

    skeletonCanvas.style.display =
        "none";


    const parent =
        cameraFeed.parentElement;

    if (!parent) {
        return;
    }


    const parentStyle =
        window.getComputedStyle(parent);

    if (
        parentStyle.position === "static"
        || !parentStyle.position
    ) {

        parent.style.position =
            "relative";
    }


    parent.appendChild(
        skeletonCanvas
    );


    skeletonContext =
        skeletonCanvas.getContext("2d");


    console.log(
        "Skeleton overlay created."
    );
}


// ============================================================
// RESIZE SKELETON CANVAS
// ============================================================

function resizeSkeletonCanvas() {

    if (
        !skeletonCanvas ||
        !cameraFeed
    ) {
        return;
    }

    if (
        !cameraFeed.videoWidth ||
        !cameraFeed.videoHeight
    ) {
        return;
    }


    if (skeletonCanvas.width !== cameraFeed.videoWidth) {
        skeletonCanvas.width = cameraFeed.videoWidth;
    }

    if (skeletonCanvas.height !== cameraFeed.videoHeight) {
        skeletonCanvas.height = cameraFeed.videoHeight;
    }
}


// ============================================================
// DRAW SKELETON
// ============================================================

function drawSkeleton(landmarks) {

    if (
        !skeletonCanvas ||
        !skeletonContext ||
        !cameraFeed
    ) {
        return;
    }


    resizeSkeletonCanvas();


    const ctx =
        skeletonContext;

    const width =
        skeletonCanvas.width;

    const height =
        skeletonCanvas.height;


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    if (
        !Array.isArray(landmarks) ||
        landmarks.length === 0
    ) {

        return;
    }


    // --------------------------------------------------------
    // DRAW CONNECTION LINES
    // --------------------------------------------------------

    ctx.lineWidth = 3;

    ctx.lineCap = "round";

    ctx.lineJoin = "round";

    ctx.strokeStyle =
        "#00ff88";


    for (
        const connection
        of POSE_CONNECTIONS
    ) {

        const start =
            landmarks[connection[0]];

        const end =
            landmarks[connection[1]];


        if (
            !start ||
            !end
        ) {
            continue;
        }


        if (
            start.visibility < 0.35 ||
            end.visibility < 0.35
        ) {
            continue;
        }


        const startX =
            start.x * width;

        const startY =
            start.y * height;

        const endX =
            end.x * width;

        const endY =
            end.y * height;


        ctx.beginPath();

        ctx.moveTo(
            startX,
            startY
        );

        ctx.lineTo(
            endX,
            endY
        );

        ctx.stroke();
    }


    // --------------------------------------------------------
    // DRAW LANDMARK POINTS
    // --------------------------------------------------------

    for (
        const landmark
        of landmarks
    ) {

        if (!landmark) {
            continue;
        }


        if (
            landmark.visibility < 0.35
        ) {
            continue;
        }


        const x =
            landmark.x * width;

        const y =
            landmark.y * height;


        // Outer circle

        ctx.beginPath();

        ctx.arc(
            x,
            y,
            5,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "#ffffff";

        ctx.fill();


        // Inner point

        ctx.beginPath();

        ctx.arc(
            x,
            y,
            3,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "#00ff88";

        ctx.fill();
    }
}


// ============================================================
// CLEAR SKELETON
// ============================================================

function clearSkeleton() {

    if (
        skeletonCanvas &&
        skeletonContext
    ) {

        skeletonContext.clearRect(
            0,
            0,
            skeletonCanvas.width,
            skeletonCanvas.height
        );
    }
}


// ============================================================
// CLOCK
// ============================================================

function updateClock() {

    if (!currentDateTime) {
        return;
    }


    const now =
        new Date();


    currentDateTime.dateTime =
        now.toISOString();


    currentDateTime.textContent =
        new Intl.DateTimeFormat(
            undefined,
            {
                month: "short",
                day: "numeric",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false
            }
        ).format(now);
}


// ============================================================
// STATUS INDICATORS
// ============================================================

function updateStatusIndicators(
    isOnline,
    isAlert,
    isCameraActive
) {

    const statusBadge =
        systemStatus?.closest(
            ".system-status"
        );


    if (statusBadge) {

        statusBadge.classList.toggle(
            "is-online",
            isOnline
        );

        statusBadge.classList.toggle(
            "is-alert",
            isAlert
        );
    }


    if (cameraStatus) {

        cameraStatus.classList.toggle(
            "is-active",
            isCameraActive
        );
    }
}


// ============================================================
// START MONITORING
// ============================================================

async function startMonitoring() {

    if (
        browserCameraActive ||
        startInProgress
    ) {
        return;
    }


    startInProgress = true;


    try {

        console.log(
            "Requesting browser camera permission..."
        );


        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {

            throw new Error(
                "Browser camera API is not available."
            );
        }


        // ----------------------------------------------------
        // ASK CAMERA PERMISSION
        // ----------------------------------------------------

        cameraStream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        facingMode: "user",
                        width: {
                            ideal: 640
                        },
                        height: {
                            ideal: 480
                        }
                    },
                    audio: false
                }
            );


        // ----------------------------------------------------
        // CONNECT CAMERA
        // ----------------------------------------------------

        if (cameraFeed) {

            cameraFeed.srcObject =
                cameraStream;

            await cameraFeed.play();
        }


        // ----------------------------------------------------
        // CREATE SKELETON OVERLAY
        // ----------------------------------------------------

        createSkeletonCanvas();

        resizeSkeletonCanvas();


        // ----------------------------------------------------
        // START BACKEND SESSION
        // ----------------------------------------------------

        const response =
            await fetch(
                "/api/start",
                {
                    method: "POST"
                }
            );


        if (!response.ok) {

            throw new Error(
                "Backend could not start monitoring."
            );
        }


        const startResult =
            await response.json();


        if (
            !startResult.success ||
            !startResult.session_id
        ) {

            throw new Error(
                "Backend session was not created."
            );
        }


        monitoringSessionId =
            startResult.session_id;


        console.log(
            "Backend session:",
            monitoringSessionId
        );


        // ----------------------------------------------------
        // CAMERA ACTIVE
        // ----------------------------------------------------

        browserCameraActive = true;


        // ----------------------------------------------------
        // UI
        // ----------------------------------------------------

        updateRunningUI();


        // ----------------------------------------------------
        // START FRAME SENDING
        // ----------------------------------------------------

        startFrameSending();


        console.log(
            "Browser camera started successfully."
        );

    } catch (error) {

        console.error(
            "Camera access/start error:",
            error
        );


        browserCameraActive = false;

        monitoringSessionId = null;


        stopFrameSending();


        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

            cameraStream = null;
        }


        if (cameraFeed) {

            cameraFeed.srcObject =
                null;
        }


        alert(
            "Camera access denied or unavailable.\n\n" +
            "Please allow camera permission and try again."
        );

    } finally {

        startInProgress = false;
    }
}


// ============================================================
// STOP MONITORING
// ============================================================

async function stopMonitoring() {

    if (
        stopInProgress ||
        (
            !browserCameraActive &&
            !cameraStream
        )
    ) {

        return;
    }


    stopInProgress = true;


    // --------------------------------------------------------
    // IMPORTANT:
    // Mark inactive FIRST.
    // This prevents new frames immediately.
    // --------------------------------------------------------

    browserCameraActive = false;


    // --------------------------------------------------------
    // Stop frame timer
    // --------------------------------------------------------

    stopFrameSending();


    // --------------------------------------------------------
    // Invalidate frontend session
    // --------------------------------------------------------

    monitoringSessionId = null;


    // --------------------------------------------------------
    // Clear skeleton
    // --------------------------------------------------------

    clearSkeleton();


    // --------------------------------------------------------
    // Stop camera
    // --------------------------------------------------------

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;
    }


    // --------------------------------------------------------
    // Clear video
    // --------------------------------------------------------

    if (cameraFeed) {

        cameraFeed.srcObject =
            null;
    }


    // --------------------------------------------------------
    // Update UI immediately
    // --------------------------------------------------------

    updateStoppedUI();


    // --------------------------------------------------------
    // Tell backend
    // --------------------------------------------------------

    try {

        await fetch(
            "/api/stop",
            {
                method: "POST"
            }
        );

    } catch (error) {

        console.error(
            "Backend stop error:",
            error
        );
    }


    stopInProgress = false;


    console.log(
        "Browser camera stopped."
    );
}


// ============================================================
// SEND CAMERA FRAME
// ============================================================

async function sendCameraFrame() {

    if (
        !browserCameraActive ||
        !monitoringSessionId
    ) {
        return;
    }

    // --------------------------------------------------------
    // Prevent overlapping requests
    // --------------------------------------------------------

    if (frameSending) {
        return;
    }


    frameSending = true;

    const requestStartedAt = performance.now();
    const requestSessionId = monitoringSessionId;

    try {

        if (
            !cameraFeed ||
            !cameraFeed.videoWidth ||
            !cameraFeed.videoHeight
        ) {
            return;
        }

        // ----------------------------------------------------
        // Resize capture canvas
        // ----------------------------------------------------

        const originalWidth =
            cameraFeed.videoWidth;

        const originalHeight =
            cameraFeed.videoHeight;

        const scale = Math.min(
            1,
            MAX_FRAME_WIDTH / originalWidth,
            MAX_FRAME_HEIGHT / originalHeight
        );
        const targetWidth = Math.max(
            1,
            Math.round(originalWidth * scale)
        );
        const targetHeight = Math.max(
            1,
            Math.round(originalHeight * scale)
        );

        if (frameCanvas.width !== targetWidth) {
            frameCanvas.width = targetWidth;
        }

        if (frameCanvas.height !== targetHeight) {
            frameCanvas.height = targetHeight;
        }


        // ----------------------------------------------------
        // Copy camera frame
        // ----------------------------------------------------

        frameContext.drawImage(
            cameraFeed,
            0,
            0,
            targetWidth,
            targetHeight
        );


        // ----------------------------------------------------
        // Convert to JPEG
        // ----------------------------------------------------

        const blob =
            await new Promise(
                resolve => {

                    frameCanvas.toBlob(
                        resolve,
                        "image/jpeg",
                        FRAME_JPEG_QUALITY
                    );
                }
            );


        if (!blob) {

            console.error(
                "Could not create camera frame blob."
            );

            return;
        }


        // ----------------------------------------------------
        // Check session again
        // ----------------------------------------------------

        if (
            !browserCameraActive ||
            monitoringSessionId !== requestSessionId
        ) {
            return;
        }


        // ----------------------------------------------------
        // Form data
        // ----------------------------------------------------

        const formData =
            new FormData();


        formData.append(
            "frame",
            blob,
            "camera.jpg"
        );


        formData.append(
            "session_id",
            requestSessionId
        );


        // ----------------------------------------------------
        // Send to backend
        // ----------------------------------------------------

        const response =
            await fetch(
                "/api/frame",
                {
                    method: "POST",
                    body: formData
                }
            );


        // ----------------------------------------------------
        // Old/stopped session
        // ----------------------------------------------------

        if (
            response.status === 409
        ) {

            return;
        }


        if (!response.ok) {

            console.error(
                "Frame API error:",
                response.status
            );

            return;
        }


        const result =
            await response.json();


        // ----------------------------------------------------
        // Ignore result if camera was stopped
        // ----------------------------------------------------

        if (
            !browserCameraActive ||
            monitoringSessionId !== requestSessionId
        ) {
            return;
        }


        // ----------------------------------------------------
        // Save latest result
        // ----------------------------------------------------

        lastDetectionResult =
            result;


        // ----------------------------------------------------
        // DRAW SKELETON
        // ----------------------------------------------------

        drawSkeleton(
            result.landmarks || []
        );


        // ----------------------------------------------------
        // UPDATE DETECTION
        // ----------------------------------------------------

        updateDetectionUI(
            result
        );


        // ----------------------------------------------------
        // FALL ALERT
        // ----------------------------------------------------

        if (
            result.detection ===
            "FALL DETECTED"
        ) {

            updateStatusIndicators(
                true,
                true,
                true
            );

        } else {

            updateStatusIndicators(
                true,
                false,
                true
            );
        }


    } catch (error) {

        // Do not spam console after stopping
        if (browserCameraActive) {

            console.error(
                "Frame sending error:",
                error
            );
        }

    } finally {

        frameSending = false;

        if (
            browserCameraActive &&
            monitoringSessionId
        ) {
            const nextFrameDelay = Math.max(
                0,
                FRAME_INTERVAL_MS
                    - (performance.now() - requestStartedAt)
            );

            frameTimer = setTimeout(
                sendCameraFrame,
                nextFrameDelay
            );
        }
    }
}


// ============================================================
// START FRAME SENDING
// ============================================================

function startFrameSending() {

    stopFrameSending();


    // Start immediately; every next frame is scheduled after this request.
    sendCameraFrame();


    console.log(
        "Browser frame sending started."
    );
}


// ============================================================
// STOP FRAME SENDING
// ============================================================

function stopFrameSending() {

    if (frameTimer) {

        clearTimeout(
            frameTimer
        );

        frameTimer = null;
    }


    console.log(
        "Browser frame sending stopped."
    );
}


// ============================================================
// RUNNING UI
// ============================================================

function updateRunningUI() {

    updateStatusIndicators(
        true,
        false,
        true
    );


    if (startBtn) {

        startBtn.disabled = true;
    }


    if (stopBtn) {

        stopBtn.disabled = false;
    }


    if (cameraPlaceholder) {

        cameraPlaceholder.style.display =
            "none";
    }


    if (cameraFeed) {

        cameraFeed.style.display =
            "block";
    }


    if (skeletonCanvas) {

        skeletonCanvas.style.display =
            "block";
    }


    if (monitoringOverlay) {

        monitoringOverlay.style.display =
            "block";
    }


    if (cameraStatus) {

        cameraStatus.textContent =
            "Camera Active";
    }


    if (systemStatus) {

        systemStatus.textContent =
            "MONITORING";
    }


    if (
        !lastDetectionResult ||
        lastDetectionResult.detection !==
            "FALL DETECTED"
    ) {

        if (detectionStatus) {

            detectionStatus.className =
                "detection-status normal";
        }


        if (statusIcon) {

            statusIcon.textContent =
                "✓";
        }


        if (detectionText) {

            detectionText.textContent =
                "MONITORING";
        }


        if (detectionDescription) {

            detectionDescription.textContent =
                "Browser camera is actively running.";
        }
    }
}


// ============================================================
// STOPPED UI
// ============================================================

function updateStoppedUI() {

    updateStatusIndicators(
        false,
        false,
        false
    );


    if (startBtn) {

        startBtn.disabled = false;
    }


    if (stopBtn) {

        stopBtn.disabled = true;
    }


    if (cameraPlaceholder) {

        cameraPlaceholder.style.display =
            "flex";
    }


    if (cameraFeed) {

        cameraFeed.style.display =
            "none";
    }


    if (skeletonCanvas) {

        skeletonCanvas.style.display =
            "none";
    }


    if (monitoringOverlay) {

        monitoringOverlay.style.display =
            "none";
    }


    if (cameraStatus) {

        cameraStatus.textContent =
            "Camera Offline";
    }


    if (systemStatus) {

        systemStatus.textContent =
            "OFFLINE";
    }


    if (detectionStatus) {

        detectionStatus.className =
            "detection-status normal";
    }


    if (statusIcon) {

        statusIcon.textContent =
            "✓";
    }


    if (detectionText) {

        detectionText.textContent =
            "SYSTEM READY";
    }


    if (detectionDescription) {

        detectionDescription.textContent =
            "Waiting for monitoring to start.";
    }


    lastDetectionResult = null;
}


// ============================================================
// UPDATE DASHBOARD STATUS
// ============================================================

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/status"
            );


        const data =
            await response.json();


        if (!data) {
            return;
        }


        // ----------------------------------------------------
        // Alert Count
        // ----------------------------------------------------

        if (alertCount) {

            alertCount.textContent =
                data.alert_count ?? 0;
        }


        // ----------------------------------------------------
        // Last Alert
        // ----------------------------------------------------

        if (lastAlert) {

            lastAlert.textContent =
                data.last_alert ||
                "—";
        }


        // ----------------------------------------------------
        // Telegram
        // ----------------------------------------------------

        if (telegramStatus) {

            telegramStatus.textContent =
                data.telegram ||
                "Ready";
        }


        // ----------------------------------------------------
        // IMPORTANT:
        // Do NOT overwrite live fall detection.
        // ----------------------------------------------------

        if (
            !browserCameraActive
        ) {

            if (systemStatus) {

                systemStatus.textContent =
                    data.status ||
                    "OFFLINE";
            }


            updateDetectionUI(
                data
            );


            updateStoppedUI();


        } else {

            if (systemStatus) {

                systemStatus.textContent =
                    "MONITORING";
            }


            // Only use backend status if no recent
            // frame result exists.

            if (!lastDetectionResult) {

                updateDetectionUI(
                    data
                );
            }


            updateStatusIndicators(
                true,
                (
                    lastDetectionResult?.detection ===
                    "FALL DETECTED"
                ),
                true
            );
        }


    } catch (error) {

        if (browserCameraActive) {

            console.error(
                "Status update error:",
                error
            );
        }
    }
}


// ============================================================
// DETECTION UI
// ============================================================

function updateDetectionUI(data) {

    if (
        !detectionStatus ||
        !statusIcon ||
        !detectionText ||
        !detectionDescription
    ) {

        return;
    }


    const detection =
        data.detection ||
        "SYSTEM READY";


    // --------------------------------------------------------
    // FALL
    // --------------------------------------------------------

    if (
        detection.includes("FALL") ||
        data.status === "ALERT"
    ) {

        detectionStatus.className =
            "detection-status danger";


        statusIcon.textContent =
            "!";


        detectionText.textContent =
            "FALL DETECTED";


        const confidence =
            data.confidence !== undefined
                ? ` ML Confidence: ${data.confidence}%.`
                : "";


        detectionDescription.textContent =
            "Alert Response Agent activated. Please check the person immediately." +
            confidence;


        return;
    }


    // --------------------------------------------------------
    // MONITORING
    // --------------------------------------------------------

    if (browserCameraActive) {

        detectionStatus.className =
            "detection-status normal";


        statusIcon.textContent =
            "✓";


        detectionText.textContent =
            "MONITORING";


        let description =
            "Browser camera is actively running.";


        if (
            data.ml_ready === false &&
            data.buffer_size !== undefined
        ) {

            description =
                `Building detection sequence... ${data.buffer_size}/30 frames.`;
        }


        detectionDescription.textContent =
            description;


        return;
    }


    // --------------------------------------------------------
    // SYSTEM READY
    // --------------------------------------------------------

    detectionStatus.className =
        "detection-status normal";


    statusIcon.textContent =
        "✓";


    detectionText.textContent =
        "SYSTEM READY";


    detectionDescription.textContent =
        "Waiting for monitoring to start.";
}


// ============================================================
// LOAD ALERT HISTORY
// ============================================================

async function loadAlertHistory() {

    try {

        const response =
            await fetch(
                "/api/alerts"
            );


        const data =
            await response.json();


        if (
            !data ||
            !data.success
        ) {

            return;
        }


        const alerts =
            Array.isArray(data.alerts)
                ? data.alerts
                : [];


        renderAlertHistory(
            alerts
        );


    } catch (error) {

        console.error(
            "Alert history error:",
            error
        );
    }
}


// ============================================================
// RENDER ALERT HISTORY
// ============================================================

function renderAlertHistory(
    alerts
) {

    if (!alertHistory) {

        return;
    }


    // --------------------------------------------------------
    // COUNT
    // --------------------------------------------------------

    if (historyCount) {

        historyCount.textContent =
            `${alerts.length} Alerts`;
    }


    // --------------------------------------------------------
    // EMPTY
    // --------------------------------------------------------

    if (
        alerts.length === 0
    ) {

        alertHistory.innerHTML = `
            <div class="empty-history">
                <span>✓</span>
                <p>No alerts detected</p>
            </div>
        `;

        return;
    }


    // --------------------------------------------------------
    // CARDS
    // --------------------------------------------------------

    alertHistory.innerHTML =
        alerts
            .map(
                alert => {

                    const incidentId =
                        alert.incident_id ||
                        "N/A";


                    const event =
                        alert.event ||
                        "FALL DETECTED";


                    const timestamp =
                        alert.timestamp ||
                        "N/A";


                    const confidence =
                        alert.confidence !== null &&
                        alert.confidence !== undefined
                            ? `${alert.confidence}%`
                            : "N/A";


                    const severity =
                        alert.severity ||
                        "N/A";


                    const telegram =
                        alert.telegram_status ||
                        "N/A";


                    // ----------------------------------------
                    // Severity
                    // ----------------------------------------

                    let severityClass =
                        "severity-low";


                    if (
                        severity.toUpperCase() ===
                        "HIGH"
                    ) {

                        severityClass =
                            "severity-high";

                    } else if (
                        severity.toUpperCase() ===
                        "MEDIUM"
                    ) {

                        severityClass =
                            "severity-medium";
                    }


                    // ----------------------------------------
                    // Telegram
                    // ----------------------------------------

                    const telegramClass =
                        telegram.toUpperCase() ===
                        "SENT"
                            ? "telegram-sent"
                            : "telegram-failed";


                    return `
                        <div class="history-item">

                            <div class="history-main">

                                <div class="history-title-row">

                                    <strong>
                                        ${escapeHtml(event)}
                                    </strong>

                                    <span class="incident-id">
                                        ${escapeHtml(incidentId)}
                                    </span>

                                </div>


                                <div class="history-details">

                                    <span>
                                        ${escapeHtml(timestamp)}
                                    </span>

                                    <span>
                                        Confidence:
                                        ${escapeHtml(confidence)}
                                    </span>

                                    <span class="${severityClass}">
                                        ${escapeHtml(severity)}
                                    </span>

                                    <span class="${telegramClass}">
                                        Telegram:
                                        ${escapeHtml(telegram)}
                                    </span>

                                </div>

                            </div>

                        </div>
                    `;
                }
            )
            .join("");
}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(value) {

    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );
}


// ============================================================
// CAMERA RESIZE
// ============================================================

if (cameraFeed) {

    cameraFeed.addEventListener(
        "loadedmetadata",
        () => {

            createSkeletonCanvas();

            resizeSkeletonCanvas();
        }
    );
}


// ============================================================
// BUTTON EVENTS
// ============================================================

if (startBtn) {

    startBtn.addEventListener(
        "click",
        startMonitoring
    );
}


if (stopBtn) {

    stopBtn.addEventListener(
        "click",
        stopMonitoring
    );
}


// ============================================================
// PAGE CLOSE
// ============================================================

window.addEventListener(
    "beforeunload",
    () => {

        browserCameraActive = false;

        stopFrameSending();


        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );
        }
    }
);


// ============================================================
// INITIALIZE DASHBOARD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    async function () {

        console.log(
            "FallDetection.AI Dashboard Loaded"
        );


        // ----------------------------------------------------
        // Clock
        // ----------------------------------------------------

        updateClock();

        setInterval(
            updateClock,
            1000
        );


        // ----------------------------------------------------
        // Initial UI
        // ----------------------------------------------------

        updateStoppedUI();


        // ----------------------------------------------------
        // Create overlay
        // ----------------------------------------------------

        createSkeletonCanvas();


        // ----------------------------------------------------
        // Status
        // ----------------------------------------------------

        await updateStatus();


        // ----------------------------------------------------
        // Alert history
        // ----------------------------------------------------

        await loadAlertHistory();


        // ----------------------------------------------------
        // Status refresh
        // ----------------------------------------------------

        setInterval(
            updateStatus,
            1000
        );


        // ----------------------------------------------------
        // Alert refresh
        // ----------------------------------------------------

        setInterval(
            loadAlertHistory,
            2000
        );

    }
);