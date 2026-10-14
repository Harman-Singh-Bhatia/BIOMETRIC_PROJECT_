/* ╔══════════════════════════════════════════════════════════════╗
   ║  BIOMM 2FA — Frontend Application (app.js)                ║
   ╚══════════════════════════════════════════════════════════════╝ */

import { FilesetResolver, HandLandmarker } from
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";

// ── State ──
let currentPage = 'landing';
let handLandmarker = null;
let cameraStream = null;
let animFrameId = null;

// Signup state
let signupUserId = '';
let signupSaved = 0;

// Login state
let loginUserId = null;
let loginTriesLeft = 3;
let loginStage1Token = null;

// Session check interval
let sessionCheckInterval = null;

// ── Gesture / Stroke (shared across signup & login) ──
const PINCH_ON = 0.45;
const PINCH_OFF = 0.88;

class OneEuroFilter {
    constructor(minCutoff = 1.6, beta = 0.6, dCutoff = 1.0) {
        this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
        this.reset();
    }
    reset() { this.xPrev = null; this.dxPrev = null; this.tPrev = null; }
    _alpha(cutoff, dt) {
        const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-6));
        return 1.0 / (1.0 + tau / Math.max(dt, 1e-6));
    }
    filter(x, t) {
        if (this.xPrev === null) {
            this.xPrev = [...x]; this.dxPrev = x.map(() => 0); this.tPrev = t;
            return [...x];
        }
        const dt = Math.max(Math.min(t - this.tPrev, 0.1), 1e-3);
        const dx = x.map((v, i) => (v - this.xPrev[i]) / dt);
        const aD = this._alpha(this.dCutoff, dt);
        const dxHat = dx.map((v, i) => aD * v + (1 - aD) * this.dxPrev[i]);
        const dxNorm = Math.sqrt(dxHat.reduce((s, v) => s + v * v, 0));
        const cutoff = this.minCutoff + this.beta * dxNorm;
        const a = this._alpha(cutoff, dt);
        const xHat = x.map((v, i) => a * v + (1 - a) * this.xPrev[i]);
        this.xPrev = [...xHat]; this.dxPrev = [...dxHat]; this.tPrev = t;
        return xHat;
    }
}

class GestureStabilizer {
    constructor(palmFrames = 5) {
        this.required = { PINCH: 2, PALM: palmFrames, FIST: 6, UNDO: 8, NONE: 2 };
        this.current = 'NONE'; this.candidate = 'NONE'; this.count = 0; this.missing = 0;
    }
    update(detected) {
        if (detected === 'LOST') {
            this.missing++;
            if (this.missing <= 8) return this.current;
            detected = 'NONE';
        } else { this.missing = 0; }
        if (detected === this.candidate) { this.count++; }
        else { this.candidate = detected; this.count = 1; }
        if (this.count >= (this.required[detected] || 3)) { this.current = detected; }
        return this.current;
    }
    reset() { this.current = 'NONE'; this.candidate = 'NONE'; this.count = 0; this.missing = 0; }
}

// ── Hand Geometry (browser-side) ──

function landmarkXY(hand, idx) { return [hand[idx].x, hand[idx].y]; }
function landmarkXYZ(hand, idx) { return [hand[idx].x, hand[idx].y, hand[idx].z]; }

function dist2(a, b) { return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2); }
function dist3(a, b) { return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2); }

function jointAngle(a, b, c) {
    const ba = a.map((v,i) => v - b[i]);
    const bc = c.map((v,i) => v - b[i]);
    const dot = ba.reduce((s, v, i) => s + v * bc[i], 0);
    const na = Math.sqrt(ba.reduce((s,v) => s + v*v, 0));
    const nb = Math.sqrt(bc.reduce((s,v) => s + v*v, 0));
    if (na * nb < 1e-8) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (na * nb)))) * (180 / Math.PI);
}

const FINGERS = {
    index: [8, 7, 6, 5], middle: [12, 11, 10, 9],
    ring: [16, 15, 14, 13], pinky: [20, 19, 18, 17]
};

function fingerExtended(hand, tip, dip, pip, mcp) {
    const aTop = jointAngle(landmarkXYZ(hand, pip), landmarkXYZ(hand, dip), landmarkXYZ(hand, tip));
    const aMid = jointAngle(landmarkXYZ(hand, mcp), landmarkXYZ(hand, pip), landmarkXYZ(hand, dip));
    const wrist = landmarkXY(hand, 0);
    const tipD = dist2(landmarkXY(hand, tip), wrist);
    const pipD = dist2(landmarkXY(hand, pip), wrist);
    return aTop >= 140 && aMid >= 140 && tipD > pipD * 1.05;
}

function fingerCurled(hand, tip, pip) {
    const wrist = landmarkXY(hand, 0);
    return dist2(landmarkXY(hand, tip), wrist) < dist2(landmarkXY(hand, pip), wrist);
}

function pinchRatio(hand) {
    const gap = dist2(landmarkXY(hand, 4), landmarkXY(hand, 8));
    const palm = dist2(landmarkXY(hand, 0), landmarkXY(hand, 9)) + 1e-6;
    return gap / palm;
}

function penPoint(hand) {
    const t = landmarkXY(hand, 4);
    const i = landmarkXY(hand, 8);
    return [(t[0] + i[0]) / 2, (t[1] + i[1]) / 2];
}

function detectGesture(hand, wasPinching) {
    const ratio = pinchRatio(hand);
    if (wasPinching && ratio < PINCH_OFF) return 'PINCH';
    const indexCurled = fingerCurled(hand, 8, 6);
    const isFist = indexCurled && fingerCurled(hand, 12, 10);
    if (!isFist && ratio < PINCH_ON) return 'PINCH';
    const ext = {};
    for (const [name, ids] of Object.entries(FINGERS)) {
        ext[name] = fingerExtended(hand, ...ids);
    }
    const count = Object.values(ext).filter(Boolean).length;
    if (count === 4) return 'PALM';
    if (count === 2 && ext.index && ext.middle) return 'UNDO';
    if (count === 0 && isFist) return 'FIST';
    return 'NONE';
}

// ── Stroke Session ──

class StrokeSession {
    constructor() {
        this.strokes = [];    // completed stroke arrays
        this.current = [];    // current stroke points
        this.filter = new OneEuroFilter();
        this.minDist = 0.003;
    }
    startStroke() { this.current = []; this.filter.reset(); }
    addPoint(xy, t) {
        const filtered = this.filter.filter(xy, t);
        if (this.current.length > 0) {
            const last = this.current[this.current.length - 1];
            if (dist2(filtered, last) < this.minDist) return filtered;
        }
        this.current.push([...filtered]);
        return filtered;
    }
    finishStroke(trim = 0) {
        let pts = trim > 0 ? this.current.slice(0, -trim) : this.current;
        if (pts.length >= 2) this.strokes.push([...pts]);
        this.current = [];
        this.filter.reset();
    }
    clear() { this.strokes = []; this.current = []; this.filter.reset(); }
    undo() {
        if (this.current.length) { this.current = []; this.filter.reset(); return true; }
        if (this.strokes.length) { this.strokes.pop(); return true; }
        return false;
    }
    allStrokes() {
        const out = [...this.strokes];
        if (this.current.length >= 2) out.push([...this.current]);
        return out;
    }
    pointCount() { return this.allStrokes().reduce((s, st) => s + st.length, 0); }
    toJSON() { return this.allStrokes(); }
}

// ── MediaPipe init ──

async function initHandLandmarker() {
    if (handLandmarker) return handLandmarker;
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
    });
    return handLandmarker;
}

// ── Camera ──

async function startCamera(videoEl) {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
    }
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });
        videoEl.srcObject = cameraStream;
        await new Promise(resolve => { videoEl.onloadeddata = resolve; });
        return true;
    } catch (err) {
        console.error("Camera error:", err);
        return false;
    }
}

function stopCamera() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
}

// ── Drawing strokes on canvas ──

function drawStrokesOnCanvas(ctx, strokes, w, h, color = '#0f172a', lineWidth = 6) {
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 0;

    // Pass 1: Subtle white/light contrast outline for visibility over webcam background
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = lineWidth + 3;
    for (const stroke of strokes) {
        if (stroke.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo((1 - stroke[0][0]) * w, stroke[0][1] * h);
        for (let i = 1; i < stroke.length; i++) {
            ctx.lineTo((1 - stroke[i][0]) * w, stroke[i][1] * h);
        }
        ctx.stroke();
    }

    // Pass 2: Bold jet black main stroke
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    for (const stroke of strokes) {
        if (stroke.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo((1 - stroke[0][0]) * w, stroke[0][1] * h);
        for (let i = 1; i < stroke.length; i++) {
            ctx.lineTo((1 - stroke[i][0]) * w, stroke[i][1] * h);
        }
        ctx.stroke();
    }
}

// ── Banner ──

function showBanner(el, text, isError = false, duration = 1500) {
    el.textContent = text;
    el.className = 'banner-overlay show' + (isError ? ' error' : '');
    setTimeout(() => { el.className = 'banner-overlay'; }, duration);
}

// ══════════════════════════════════════════
//   PAGE NAVIGATION
// ══════════════════════════════════════════

window.showPage = function(page) {
    // Clean up camera when leaving
    if (currentPage !== page) stopCamera();

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const el = document.getElementById('page-' + (page === 'safe' ? 'safe' : page));
    if (el) el.classList.add('active');
    currentPage = page;

    if (page === 'safe') loadSafeArea();
};

// ══════════════════════════════════════════
//   SIGNUP FLOW
// ══════════════════════════════════════════

function setSignupStep(n) {
    // Update stepper
    document.querySelectorAll('#signup-stepper .step').forEach((s, i) => {
        s.classList.remove('active', 'done');
        if (i + 1 < n) s.classList.add('done');
        else if (i + 1 === n) s.classList.add('active');
    });
    // Show step content
    ['signup-step1', 'signup-step1b', 'signup-step2', 'signup-step3'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });
}

window.signupStart = async function() {
    const uid = document.getElementById('signup-uid').value.trim();
    if (!uid) {
        document.getElementById('signup-step1-status').textContent = 'Please enter a user ID';
        document.getElementById('signup-step1-status').className = 'status-text error-text';
        return;
    }
    const statusEl = document.getElementById('signup-step1-status');
    statusEl.textContent = 'Linking fingerprint...';
    statusEl.className = 'status-text';

    try {
        const resp = await fetch('/api/signup/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: uid })
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
            statusEl.textContent = data.detail || data.reason || 'Signup failed';
            statusEl.className = 'status-text error-text';
            return;
        }
        signupUserId = data.user_id;
        signupSaved = 0;
        document.getElementById('signup-linked-file').textContent = `Linked: ${data.linked_file}`;
        setSignupStep(1);
        document.getElementById('signup-step1b').classList.add('active');
    } catch (err) {
        statusEl.textContent = 'Network error: ' + err.message;
        statusEl.className = 'status-text error-text';
    }
};

window.signupGoToSignature = async function() {
    setSignupStep(2);
    document.getElementById('signup-step2').classList.add('active');
    await startSignatureCapture('signup');
};

async function startSignatureCapture(mode) {
    const prefix = mode; // 'signup' or 'login'
    const videoEl = document.getElementById(prefix + '-video');
    const canvasEl = document.getElementById(prefix + '-canvas');
    const cursorEl = document.getElementById(prefix + '-cursor');
    const bannerEl = document.getElementById(prefix + '-banner');
    const statusEl = document.getElementById(prefix + '-status');
    const pinchFill = document.getElementById(prefix + '-pinch-fill');
    const camInfo = document.getElementById(prefix + '-cam-info');

    // Init MediaPipe
    try {
        await initHandLandmarker();
    } catch (err) {
        statusEl.textContent = 'Failed to load hand tracking: ' + err.message;
        return;
    }

    // Start camera
    const camOk = await startCamera(videoEl);
    if (!camOk) {
        statusEl.textContent = '⚠ Camera access denied or not available';
        statusEl.style.color = 'var(--red)';
        return;
    }

    // Set canvas size
    const rect = canvasEl.parentElement.getBoundingClientRect();
    canvasEl.width = videoEl.videoWidth || rect.width;
    canvasEl.height = videoEl.videoHeight || rect.height;
    const ctx = canvasEl.getContext('2d');

    const session = new StrokeSession();
    const stabilizer = new GestureStabilizer(5);
    let prevGesture = 'NONE';
    let palmCooldown = 0;
    let lastFps = 0;
    let lastTime = performance.now();
    let submitting = false;

    let lockedPenPos = null;

    function loop() {
        if (!cameraStream) return;
        animFrameId = requestAnimationFrame(loop);

        const now = performance.now();
        const dt = now - lastTime;
        if (dt < 25) return; // ~40fps max
        const fps = 1000 / dt;
        lastFps = lastFps === 0 ? fps : 0.9 * lastFps + 0.1 * fps;
        lastTime = now;

        // Detect hands
        let detectedHands = [];
        try {
            const results = handLandmarker.detectForVideo(videoEl, now);
            if (results.landmarks && results.landmarks.length > 0) {
                detectedHands = results.landmarks;
            }
        } catch (e) { /* ignore detection errors */ }

        const handCount = detectedHands.length;

        // Handle multi-hand detection lock
        if (handCount > 1) {
            statusEl.textContent = '⚠ Multiple hands detected — only one hand allowed while signing';
            statusEl.style.color = 'var(--red)';
            cursorEl.style.display = 'none';

            // Pause drawing gesture input while multiple hands are present
            prevGesture = 'NONE';
            stabilizer.reset();

            drawStrokesOnCanvas(ctx, session.allStrokes(), canvasEl.width, canvasEl.height);
            camInfo.textContent = `hands: ${handCount} [PAUSED] | fps: ${lastFps.toFixed(0)}`;
            return;
        }

        let hand = null;
        if (handCount === 1) {
            hand = detectedHands[0];
            const currentPen = penPoint(hand);
            if (lockedPenPos && session.current.length > 0) {
                const d = dist2(currentPen, lockedPenPos);
                // Allow fast rapid hand motion across screen up to 0.85 normalized distance per frame
                if (d > 0.85) {
                    hand = null;
                } else {
                    lockedPenPos = currentPen;
                }
            } else {
                lockedPenPos = currentPen;
            }
        } else {
            lockedPenPos = null;
        }

        const ratio = hand ? pinchRatio(hand) : 1.0;
        const raw = hand ? detectGesture(hand, prevGesture === 'PINCH') : 'LOST';
        const gesture = stabilizer.update(raw);

        // Reset status color when single hand is restored
        if (handCount <= 1 && statusEl.textContent.includes('Multiple hands')) {
            statusEl.textContent = 'Pinch thumb + index to draw · Palm to submit';
            statusEl.style.color = '';
        }

        // Pinch meter
        const meterPct = Math.max(0, Math.min(100, (1 - ratio / 1.2) * 100));
        pinchFill.style.width = meterPct + '%';
        pinchFill.className = 'meter-fill' + (gesture === 'PINCH' ? ' pinching' : '');

        // Cursor
        if (hand) {
            const pp = penPoint(hand);
            const wrapRect = canvasEl.parentElement.getBoundingClientRect();
            cursorEl.style.left = ((1 - pp[0]) * wrapRect.width) + 'px';
            cursorEl.style.top = (pp[1] * wrapRect.height) + 'px';
            cursorEl.style.display = 'block';
            cursorEl.className = 'pen-cursor' + (gesture === 'PINCH' ? ' pinching' : '');
        } else {
            cursorEl.style.display = 'none';
        }

        // Gesture handling
        const started = gesture !== prevGesture;
        const t = now / 1000;

        if (gesture === 'PINCH') {
            if (started) session.startStroke();
            if (hand) session.addPoint(penPoint(hand), t);
        } else if (prevGesture === 'PINCH') {
            session.finishStroke(gesture === 'PALM' ? 6 : 0);
        }

        if (gesture === 'PALM' && started && palmCooldown <= 0 && !submitting) {
            palmCooldown = 25;
            onPalmSubmit(mode, session, bannerEl, statusEl);
        } else if (gesture === 'FIST' && started) {
            session.clear();
            statusEl.textContent = 'Cleared';
        } else if (gesture === 'UNDO' && started) {
            if (session.undo()) statusEl.textContent = 'Last stroke removed';
        }

        if (palmCooldown > 0) palmCooldown--;
        prevGesture = gesture;

        // Draw strokes
        drawStrokesOnCanvas(ctx, session.allStrokes(), canvasEl.width, canvasEl.height);

        // Info
        camInfo.textContent = `gesture: ${gesture} | fps: ${lastFps.toFixed(0)}`;

        // Update progress (signup only)
        if (mode === 'signup') {
            document.getElementById('signup-progress-text').textContent = `SAVED ${signupSaved}/5`;
            document.getElementById('signup-progress-fill').style.width = (signupSaved / 5 * 100) + '%';
        }
    }

    animFrameId = requestAnimationFrame(loop);

    async function onPalmSubmit(mode, session, bannerEl, statusEl) {
        const strokes = session.allStrokes();
        if (strokes.length === 0 || session.pointCount() < 12) {
            statusEl.textContent = 'Draw a signature first';
            return;
        }
        submitting = true;
        statusEl.textContent = 'Processing...';

        if (mode === 'signup') {
            try {
                const resp = await fetch('/api/signup/signature', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: signupUserId, strokes: strokes })
                });
                const data = await resp.json();
                if (data.ok) {
                    signupSaved = data.saved;
                    showBanner(bannerEl, `SAVED ${data.saved}/${data.total}`);
                    session.clear();
                    statusEl.textContent = data.complete
                        ? 'Enrollment complete!'
                        : `Saved ${data.saved}/${data.total} — draw again`;

                    if (data.complete) {
                        setTimeout(() => {
                            stopCamera();
                            setSignupStep(3);
                            document.getElementById('signup-step3').classList.add('active');
                            document.getElementById('signup-complete-info').textContent =
                                `User "${signupUserId}" is now ACTIVE. Threshold: ${data.threshold}, Consistency: ${data.consistency}`;
                        }, 1200);
                    }
                } else {
                    showBanner(bannerEl, 'DRAW AGAIN', true);
                    session.clear();
                    statusEl.textContent = data.reason || 'Too different — try again';
                }
            } catch (err) {
                statusEl.textContent = 'Error: ' + err.message;
            }
        } else {
            // Login signature
            try {
                const resp = await fetch('/api/login/signature', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: loginUserId, strokes: strokes })
                });
                const data = await resp.json();
                if (data.ok) {
                    loginStage1Token = data.stage1_token;
                    showBanner(bannerEl, 'MATCH');
                    statusEl.textContent = `Signature accepted — score ${data.score}`;
                    setTimeout(() => {
                        stopCamera();
                        startFingerprintScan(data);
                    }, 1200);
                } else {
                    loginTriesLeft--;
                    showBanner(bannerEl, 'NO MATCH', true);
                    session.clear();
                    document.getElementById('login-tries').textContent = `${loginTriesLeft} tries left`;
                    statusEl.textContent = data.reason || 'Signature did not match';
                    if (loginTriesLeft <= 0) {
                        setTimeout(() => {
                            stopCamera();
                            showFingerprintResult(false, 'ACCESS DENIED', 'Maximum attempts exceeded');
                        }, 1500);
                    }
                }
            } catch (err) {
                statusEl.textContent = 'Error: ' + err.message;
            }
        }
        submitting = false;
    }
}

// ══════════════════════════════════════════
//   LOGIN FLOW
// ══════════════════════════════════════════

function setLoginStep(n) {
    document.querySelectorAll('#login-stepper .step').forEach((s, i) => {
        s.classList.remove('active', 'done');
        if (i + 1 < n) s.classList.add('done');
        else if (i + 1 === n) s.classList.add('active');
    });
    ['login-step0', 'login-step1', 'login-step2', 'login-step2b'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });
}

window.loginStart = async function() {
    const uid = document.getElementById('login-uid').value.trim() || null;
    loginUserId = uid;
    loginTriesLeft = 3;
    loginStage1Token = null;
    document.getElementById('login-tries').textContent = '3 tries left';

    const modeLabel = document.getElementById('login-mode-label');
    modeLabel.textContent = uid ? `AIR SIGNATURE · VERIFY · ${uid}` : 'AIR SIGNATURE · IDENTIFY';

    setLoginStep(1);
    document.getElementById('login-step1').classList.add('active');
    await startSignatureCapture('login');
};

// ══════════════════════════════════════════
//   FINGERPRINT SCANNING ANIMATION
// ══════════════════════════════════════════

async function startFingerprintScan(sigData) {
    setLoginStep(2);
    document.getElementById('login-step2').classList.add('active');
    document.getElementById('login-back-btn').style.display = 'none';

    const fpCard = document.querySelector('.fp-card');
    fpCard.className = 'glass-card card-lg centered-card fp-card';

    const statusText = document.getElementById('fp-status-text');
    const progressFill = document.getElementById('fp-progress-fill');
    const pointsCount = document.getElementById('fp-points-count');
    const laserEl = document.getElementById('fp-laser');

    // Reset
    laserEl.style.display = '';
    progressFill.style.width = '0%';
    pointsCount.textContent = '0 points';

    // Status sequence
    const messages = [
        'Signature accepted',
        'Fetching linked fingerprint...',
        'Scanning fingerprint...',
        'Extracting minutiae...',
        'Matching with enrolled template...'
    ];

    // Start the API call immediately
    const fpPromise = fetch('/api/login/fingerprint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ stage1_token: loginStage1Token })
    }).then(r => r.json()).catch(err => ({ granted: false, reason: err.message }));

    const minDuration = 2500;
    const startTime = Date.now();

    // Animate messages
    for (let i = 0; i < messages.length; i++) {
        statusText.textContent = messages[i];
        progressFill.style.width = ((i + 1) / messages.length * 80) + '%';

        // Tick up points counter
        const targetPts = Math.floor(Math.random() * 8 + 5) * (i + 1);
        pointsCount.textContent = `${targetPts} points`;

        await sleep(500);
    }

    // Wait for both the API and minimum duration
    const [fpResult] = await Promise.all([
        fpPromise,
        sleep(Math.max(0, minDuration - (Date.now() - startTime)))
    ]);

    progressFill.style.width = '100%';
    pointsCount.textContent = `${Math.floor(Math.random() * 10 + 25)} points`;

    await sleep(300);

    if (fpResult.granted) {
        // SUCCESS
        fpCard.classList.add('success');
        laserEl.style.display = 'none';
        statusText.textContent = 'FINGERPRINT VERIFIED';
        await sleep(800);
        statusText.textContent = 'ACCESS GRANTED';
        await sleep(1200);
        showPage('safe');
    } else {
        // FAILURE
        fpCard.classList.add('failure');
        laserEl.style.display = 'none';
        statusText.textContent = 'FINGERPRINT MISMATCH';
        await sleep(800);
        showFingerprintResult(false, 'ACCESS DENIED', fpResult.reason || 'Verification failed');
    }
}

function showFingerprintResult(success, title, detail) {
    setLoginStep(2);
    document.getElementById('login-step2').classList.remove('active');
    document.getElementById('login-step2b').classList.add('active');

    const card = document.getElementById('fp-result-card');
    const iconEl = document.getElementById('fp-result-icon');
    const titleEl = document.getElementById('fp-result-title');
    const detailEl = document.getElementById('fp-result-detail');
    const retryBtn = document.getElementById('fp-retry-btn');

    titleEl.textContent = title;
    detailEl.textContent = detail;

    if (success) {
        card.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        titleEl.className = 'mono success-text';
        iconEl.innerHTML = `<div class="check-anim"><svg viewBox="0 0 64 64"><circle class="check-circle" cx="32" cy="32" r="28" fill="none" stroke="var(--success)" stroke-width="2.5"/><path class="check-mark" d="M20 32l8 8 16-16" fill="none" stroke="var(--success)" stroke-width="3" stroke-linecap="round"/></svg></div>`;
        retryBtn.style.display = 'none';
    } else {
        card.style.borderColor = 'rgba(255,61,90,0.3)';
        titleEl.className = 'mono error-text';
        iconEl.innerHTML = `<svg viewBox="0 0 64 64" width="72" height="72"><circle cx="32" cy="32" r="28" fill="none" stroke="var(--red)" stroke-width="2.5"/><line x1="22" y1="22" x2="42" y2="42" stroke="var(--red)" stroke-width="3" stroke-linecap="round"/><line x1="42" y1="22" x2="22" y2="42" stroke="var(--red)" stroke-width="3" stroke-linecap="round"/></svg>`;
        retryBtn.style.display = '';
    }
    document.getElementById('login-back-btn').style.display = '';
}

// ══════════════════════════════════════════
//   SAFE AREA
// ══════════════════════════════════════════

async function loadSafeArea() {
    try {
        const resp = await fetch('/api/me', { credentials: 'include' });
        if (!resp.ok) {
            window.location.href = '/';
            return;
        }
        const data = await resp.json();
        document.getElementById('safe-user-id').textContent = data.user_id;
        document.getElementById('safe-welcome').textContent = `Access granted. Welcome, ${data.user_id}.`;
        document.getElementById('safe-login-time').textContent = `Login: ${data.login_time}`;
        document.getElementById('safe-sig-score').textContent = data.sig_score?.toFixed(4) || '—';
        document.getElementById('safe-fp-score').textContent = data.fp_score?.toFixed(4) || '—';
        document.getElementById('safe-fp-pairs').textContent = data.fp_pairs || '—';

        // Activity
        const actEl = document.getElementById('safe-activity');
        if (data.recent_activity && data.recent_activity.length) {
            actEl.innerHTML = data.recent_activity.map(r =>
                `<div class="activity-row"><span class="ev">${r.event || ''}</span><span class="tm">${r.time || ''}</span><span>${r.user || ''}</span></div>`
            ).join('');
        } else {
            actEl.textContent = 'No recent activity';
        }

        // Start session expiry checker
        if (sessionCheckInterval) clearInterval(sessionCheckInterval);
        sessionCheckInterval = setInterval(async () => {
            try {
                const r = await fetch('/api/me', { credentials: 'include' });
                if (!r.ok) {
                    clearInterval(sessionCheckInterval);
                    document.getElementById('session-expired-overlay').style.display = 'flex';
                }
            } catch { /* ignore */ }
        }, 30000);

    } catch (err) {
        console.error('Safe area load failed:', err);
        window.location.href = '/';
    }
}

window.doLogout = async function() {
    if (sessionCheckInterval) clearInterval(sessionCheckInterval);
    try {
        await fetch('/api/logout', { method: 'POST', credentials: 'include' });
    } catch { /* ignore */ }
    window.location.href = '/';
};

// ── Utility ──
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init on page load ──
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're at /safe-area URL
    if (window.location.pathname === '/safe-area') {
        showPage('safe');
    }
});
