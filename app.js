/**
 * app.js — ボーイングカメラ メインロジック
 */

// ── MediaPipe CDN ──
const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const MODEL_FULL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
const MODEL_LITE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';

// ── Landmark indices ──
const LM = {
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  RIGHT_ELBOW: 14,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
};

// ── 右腕描画用の接続定義 ──
const RIGHT_ARM_CONNECTIONS = [
  [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
  [LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
];
const TORSO_CONNECTIONS = [
  [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
  [LM.LEFT_SHOULDER, LM.LEFT_HIP],
  [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  [LM.LEFT_HIP, LM.RIGHT_HIP],
];
const HEAD_CONNECTIONS = [
  [LM.LEFT_EAR, LM.LEFT_SHOULDER],
  [LM.RIGHT_EAR, LM.RIGHT_SHOULDER],
];

// ── DOM ──
const $ = id => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
const startBtn = $('start-btn');
const startIcon = $('start-icon');
const startText = $('start-text');
const cameraStatus = $('camera-status');
const fpsBadge = $('fps-badge');
const delegateBadge = $('delegate-badge');
const cameraToggleBtn = $('camera-toggle-btn');
const cameraToggleIcon = $('camera-toggle-icon');
const calibrateBtn = $('calibrate-btn');
const diagnoseBtn = $('diagnose-btn');
const diagnoseCountdown = $('diagnose-countdown');
const ringProgress = $('ring-progress');
const countdownNum = $('countdown-num');
const diagnoseModal = $('diagnose-modal');
const diagnoseResults = $('diagnose-results');
const diagnoseCloseBtn = $('diagnose-close-btn');

const cardStraightness = $('card-straightness');
const cardDistribution = $('card-distribution');
const cardElbow = $('card-elbow');
const cardShoulder = $('card-shoulder');
const valStraightness = $('val-straightness');
const valDistribution = $('val-distribution');
const valElbow = $('val-elbow');
const valShoulder = $('val-shoulder');
const subStraightness = $('sub-straightness');
const subElbow = $('sub-elbow');
const subShoulder = $('sub-shoulder');
const distTip = $('dist-tip');
const distMid = $('dist-mid');
const distFrog = $('dist-frog');
const adviceText = $('advice-text');

// ── State ──
const state = {
  isRunning: false,
  facingMode: 'user',     // 'user' = 内カメ, 'environment' = 外カメ
  wristTrail: [],         // [{x, y, t}] 直近60フレーム
  bowDistribution: { tip: 0, middle: 0, frog: 0 },
  bowDistResetTime: 0,
  baseShoulderEarDist: null,
  lastShoulderEarDist: null,

  // smoothed values
  smoothStraightness: null,
  smoothElbowHeight: null,
  smoothShoulderTension: null,

  // 現在のメトリクス
  currentMetrics: {},

  // 診断モード
  diagnosing: false,
  diagnoseStartTime: 0,
  diagnoseTimerId: null,
  diagnoseLog: [],        // 各フレームのメトリクスを蓄積
};

let poseLandmarker = null;
let stream = null;
let lastVideoTime = -1;
let frameCount = 0;
let fpsTime = 0;
let currentFps = 0;
let delegate = '';

// ── Init ──
startBtn.addEventListener('click', toggleRunning);
cameraToggleBtn.addEventListener('click', toggleCamera);
calibrateBtn.addEventListener('click', calibrateShoulder);
diagnoseBtn.addEventListener('click', toggleDiagnose);
diagnoseCloseBtn.addEventListener('click', () => diagnoseModal.classList.add('hidden'));

async function toggleRunning() {
  if (state.isRunning) {
    stop();
  } else {
    await start();
  }
}

async function start() {
  startBtn.disabled = true;
  cameraStatus.textContent = 'モデルを読み込み中...';
  cameraStatus.classList.remove('hidden');

  try {
    if (!poseLandmarker) {
      await initPoseLandmarker();
    }
    await startCamera();
    state.isRunning = true;
    state.baseShoulderEarDist = null;
    state.bowDistribution = { tip: 0, middle: 0, frog: 0 };
    state.bowDistResetTime = performance.now();
    state.wristTrail = [];
    startBtn.classList.add('running');
    startIcon.textContent = '⏹';
    startText.textContent = '停止';
    calibrateBtn.classList.remove('hidden');
    cameraStatus.textContent = '🎯ボタンで構えた状態を記録';
    detectLoop();
  } catch (e) {
    cameraStatus.textContent = 'エラー: ' + e.message;
    console.error(e);
  }
  startBtn.disabled = false;
}

function stop() {
  state.isRunning = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.classList.remove('running');
  startIcon.textContent = '▶';
  startText.textContent = '開始';
  calibrateBtn.classList.add('hidden');
  cameraStatus.textContent = '開始ボタンを押してください';
  cameraStatus.classList.remove('hidden');
}

// ── MediaPipe ──
async function initPoseLandmarker() {
  const { FilesetResolver, PoseLandmarker } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18'
  );

  // GPU優先
  try {
    cameraStatus.textContent = 'GPU初期化中...';
    const vision = await FilesetResolver.forVisionTasks(VISION_CDN);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_FULL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.15,
      minPosePresenceConfidence: 0.15,
      minTrackingConfidence: 0.15,
    });
    delegate = 'GPU';
  } catch (e) {
    // CPU fallback
    console.warn('GPU failed, falling back to CPU:', e);
    cameraStatus.textContent = 'CPUモードで再試行中...';
    const vision = await FilesetResolver.forVisionTasks(VISION_CDN);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_LITE, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    delegate = 'CPU';
  }
  delegateBadge.textContent = delegate;
}

// ── Camera Toggle ──
async function toggleCamera() {
  state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
  updateCameraIcon();
  if (state.isRunning) {
    await startCamera();
  }
}

// ── Calibration ──
function calibrateShoulder() {
  if (state.lastShoulderEarDist && state.lastShoulderEarDist > 0) {
    state.baseShoulderEarDist = state.lastShoulderEarDist;
    calibrateBtn.classList.add('calibrated');
    cameraStatus.textContent = '✅ キャリブレーション完了';
    cameraStatus.classList.remove('hidden');
    setTimeout(() => {
      if (state.isRunning) cameraStatus.classList.add('hidden');
    }, 1500);
  }
}

function updateCameraIcon() {
  const isInner = state.facingMode === 'user';
  cameraToggleIcon.textContent = isInner ? '🤳' : '📷';
  // ミラー: 内カメのみ
  video.classList.toggle('mirror', isInner);
  overlay.classList.toggle('mirror', isInner);
}

// ── Camera ──
async function startCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }
  cameraStatus.textContent = 'カメラを起動中...';
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: state.facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  updateCameraIcon();

  await new Promise(resolve => {
    video.addEventListener('loadeddata', resolve, { once: true });
  });
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

// ── Detect Loop ──
function detectLoop() {
  if (!state.isRunning) return;
  requestAnimationFrame(detectLoop);
  if (!poseLandmarker || video.readyState < 2) return;

  const t = video.currentTime;
  if (t === lastVideoTime) return;
  lastVideoTime = t;

  const ts = performance.now();
  const result = poseLandmarker.detectForVideo(video, ts);
  frameCount++;

  // FPS
  if (ts - fpsTime >= 1000) {
    currentFps = frameCount;
    frameCount = 0;
    fpsTime = ts;
    fpsBadge.textContent = currentFps + ' FPS';
  }

  if (result.landmarks && result.landmarks.length > 0) {
    const lm = result.landmarks[0];
    processFrame(lm, ts);
    drawOverlay(lm);
  } else {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

// ── Process Frame ──
function processFrame(lm, ts) {
  const rShoulder = lm[LM.RIGHT_SHOULDER];
  const rElbow = lm[LM.RIGHT_ELBOW];
  const rWrist = lm[LM.RIGHT_WRIST];
  const lShoulder = lm[LM.LEFT_SHOULDER];
  const rEar = lm[LM.RIGHT_EAR];
  const lEar = lm[LM.LEFT_EAR];
  const rHip = lm[LM.RIGHT_HIP];

  // 手首軌跡の蓄積（直近60フレーム）
  state.wristTrail.push({ x: rWrist.x, y: rWrist.y, t: ts });
  if (state.wristTrail.length > 60) {
    state.wristTrail.shift();
  }

  // M1: 弓の直線性
  const straightness = BowMetrics.computeBowStraightness(state.wristTrail);
  if (straightness.score !== null) {
    state.smoothStraightness = BowMetrics.ema(state.smoothStraightness, straightness.score, 0.15);
  }

  // M2: 弓の配分
  const bowZone = BowMetrics.computeBowZone(rShoulder, rElbow, rWrist);
  state.bowDistribution[bowZone.zone]++;
  // 10秒ごとにリセット
  if (ts - state.bowDistResetTime > 10000) {
    state.bowDistribution = { tip: 0, middle: 0, frog: 0 };
    state.bowDistResetTime = ts;
    state.bowDistribution[bowZone.zone]++;
  }
  const distribution = BowMetrics.computeDistributionPercent(state.bowDistribution);

  // M3: 肘の高さ
  const elbowHeight = BowMetrics.computeElbowHeight(rShoulder, rElbow, rHip, null);
  state.smoothElbowHeight = BowMetrics.ema(
    state.smoothElbowHeight, elbowHeight.relativeHeight, 0.1
  );

  // M4: 肩の緊張（手動キャリブレーション）
  const shoulderTension = BowMetrics.computeShoulderTension(
    rShoulder, rEar, lShoulder, lEar, state.baseShoulderEarDist
  );
  // 最新の肩-耳距離を保持（キャリブレーションボタン用）
  state.lastShoulderEarDist = shoulderTension.currentDist;

  // スムージング後のスコアからステータスを再計算（Bug #1修正）
  const smoothedScore = state.smoothStraightness;
  const straightnessStatus = smoothedScore !== null
    ? (smoothedScore >= 85 ? 'good' : smoothedScore >= 65 ? 'warn' : 'bad')
    : 'good';

  // 診断モード: メトリクスを蓄積
  if (state.diagnosing) {
    state.diagnoseLog.push({
      straightness: smoothedScore !== null ? Math.round(smoothedScore) : null,
      straightnessStatus: straightnessStatus,
      elbowHeight: state.smoothElbowHeight !== null ? Math.round(state.smoothElbowHeight * 100) / 100 : elbowHeight.relativeHeight,
      elbowStatus: elbowHeight.status,
      elbowLabel: elbowHeight.label,
      shoulderTension: shoulderTension.tension,
      shoulderStatus: shoulderTension.status,
      shoulderLabel: shoulderTension.label,
      bowZone: bowZone.zone,
      extensionRatio: bowZone.extensionRatio,
    });
  }

  // メトリクスを保存
  const smoothedRound = smoothedScore !== null ? Math.round(smoothedScore) : null;
  state.currentMetrics = {
    straightness: {
      score: smoothedRound,
      status: straightnessStatus,
    },
    distribution,
    bowZone: bowZone,
    elbow: {
      ...elbowHeight,
      relativeHeight: state.smoothElbowHeight !== null
        ? Math.round(state.smoothElbowHeight * 100) / 100
        : elbowHeight.relativeHeight,
    },
    shoulder: state.baseShoulderEarDist === null
      ? { tension: 0, status: 'good', label: 'キャリブレーション待ち', currentDist: shoulderTension.currentDist }
      : shoulderTension,
  };

  updateUI(state.currentMetrics);
}

// ── Update UI ──
function updateUI(m) {
  // M1: 直線性
  if (m.straightness.score !== null) {
    valStraightness.textContent = m.straightness.score + '%';
    subStraightness.textContent =
      m.straightness.status === 'good' ? 'まっすぐ' :
      m.straightness.status === 'warn' ? 'やや曲がり' : '曲がっています';
  }
  setCardStatus(cardStraightness, m.straightness.status);

  // M2: 配分
  valDistribution.textContent = m.distribution.label;
  distTip.style.width = m.distribution.tip + '%';
  distMid.style.width = m.distribution.middle + '%';
  distFrog.style.width = m.distribution.frog + '%';

  // M3: 肘
  valElbow.textContent = m.elbow.label;
  subElbow.textContent = '相対高さ ' + m.elbow.relativeHeight + '（※弦により変動）';
  setCardStatus(cardElbow, m.elbow.status);

  // M4: 肩
  valShoulder.textContent = m.shoulder.label;
  if (m.shoulder.tension > 0) {
    subShoulder.textContent = '緊張度 ' + m.shoulder.tension + '%';
  } else {
    subShoulder.textContent = '';
  }
  setCardStatus(cardShoulder, m.shoulder.status);

  // アドバイス
  adviceText.textContent = BowMetrics.generateAdvice(m);
}

function setCardStatus(card, status) {
  card.classList.remove('good', 'warn', 'bad');
  if (status) card.classList.add(status);
}

// ── Draw Overlay ──
function drawOverlay(lm) {
  const w = overlay.width;
  const h = overlay.height;
  ctx.clearRect(0, 0, w, h);

  // 体幹の接続線（薄く）
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
  for (const [si, ei] of [...TORSO_CONNECTIONS, ...HEAD_CONNECTIONS]) {
    const s = lm[si], e = lm[ei];
    if ((s.visibility || 0) < 0.3 || (e.visibility || 0) < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(s.x * w, s.y * h);
    ctx.lineTo(e.x * w, e.y * h);
    ctx.stroke();
  }

  // 右腕の接続線（ステータスに応じた色）
  const armStatus = state.currentMetrics.elbow
    ? state.currentMetrics.elbow.status
    : 'good';
  const armColor =
    armStatus === 'good' ? 'rgba(46, 204, 113, 0.9)' :
    armStatus === 'warn' ? 'rgba(241, 196, 15, 0.9)' :
    'rgba(231, 76, 60, 0.9)';

  ctx.lineWidth = 3;
  ctx.strokeStyle = armColor;
  for (const [si, ei] of RIGHT_ARM_CONNECTIONS) {
    const s = lm[si], e = lm[ei];
    if ((s.visibility || 0) < 0.3 || (e.visibility || 0) < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(s.x * w, s.y * h);
    ctx.lineTo(e.x * w, e.y * h);
    ctx.stroke();
  }

  // 関連ジョイントの点
  const highlightIdx = new Set([
    LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
    LM.LEFT_SHOULDER, LM.LEFT_EAR, LM.RIGHT_EAR,
  ]);
  for (let i = 0; i < lm.length; i++) {
    if ((lm[i].visibility || 0) < 0.3) continue;
    const isHL = highlightIdx.has(i);
    ctx.beginPath();
    ctx.arc(lm[i].x * w, lm[i].y * h, isHL ? 4 : 2, 0, Math.PI * 2);
    ctx.fillStyle = isHL ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.2)';
    ctx.fill();
    if (isHL) {
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // 弓軌跡ライン
  drawBowTrail(w, h);
}

function drawBowTrail(w, h) {
  const trail = state.wristTrail;
  if (trail.length < 2) return;

  const straightStatus = state.currentMetrics.straightness
    ? state.currentMetrics.straightness.status
    : 'good';

  for (let i = 1; i < trail.length; i++) {
    const age = (trail.length - i) / trail.length;  // 0=新しい, 1=古い
    const alpha = 0.8 * (1 - age * 0.8);
    const lineW = 3 * (1 - age * 0.6);

    let r, g, b;
    if (straightStatus === 'good') { r = 46; g = 204; b = 113; }
    else if (straightStatus === 'warn') { r = 241; g = 196; b = 15; }
    else { r = 231; g = 76; b = 60; }

    ctx.beginPath();
    ctx.moveTo(trail[i - 1].x * w, trail[i - 1].y * h);
    ctx.lineTo(trail[i].x * w, trail[i].y * h);
    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.lineWidth = lineW;
    ctx.lineCap = 'round';
    ctx.stroke();
  }
}

// ── 15秒診断モード ──
const DIAGNOSE_DURATION = 15;
const RING_CIRCUMFERENCE = 2 * Math.PI * 44; // r=44

function toggleDiagnose() {
  if (state.diagnosing) {
    cancelDiagnose();
  } else {
    startDiagnose();
  }
}

async function startDiagnose() {
  // カメラが動いていなければ先に起動
  if (!state.isRunning) {
    await start();
    if (!state.isRunning) return; // 起動失敗
  }

  state.diagnosing = true;
  state.diagnoseLog = [];
  state.diagnoseStartTime = performance.now();

  diagnoseBtn.classList.add('running');
  diagnoseCountdown.classList.remove('hidden');
  countdownNum.textContent = DIAGNOSE_DURATION;
  ringProgress.style.strokeDasharray = RING_CIRCUMFERENCE;
  ringProgress.style.strokeDashoffset = '0';

  // 1秒ごとにカウントダウン更新
  let remaining = DIAGNOSE_DURATION;
  state.diagnoseTimerId = setInterval(() => {
    remaining--;
    if (remaining <= 0) {
      finishDiagnose();
    } else {
      countdownNum.textContent = remaining;
      const progress = (DIAGNOSE_DURATION - remaining) / DIAGNOSE_DURATION;
      ringProgress.style.strokeDashoffset = (RING_CIRCUMFERENCE * progress).toFixed(1);
    }
  }, 1000);
}

function cancelDiagnose() {
  state.diagnosing = false;
  if (state.diagnoseTimerId) {
    clearInterval(state.diagnoseTimerId);
    state.diagnoseTimerId = null;
  }
  diagnoseBtn.classList.remove('running');
  diagnoseCountdown.classList.add('hidden');
}

function finishDiagnose() {
  cancelDiagnose();
  showDiagnoseReport(state.diagnoseLog);
}

function showDiagnoseReport(log) {
  if (log.length === 0) {
    diagnoseResults.innerHTML = '<p style="text-align:center;color:var(--text-dim)">ポーズが検出できませんでした。<br>カメラに全身が映るように調整してください。</p>';
    diagnoseModal.classList.remove('hidden');
    return;
  }

  // 集計
  const validStraightness = log.filter(l => l.straightness !== null).map(l => l.straightness);
  const avgStraightness = validStraightness.length > 0
    ? Math.round(validStraightness.reduce((a, b) => a + b, 0) / validStraightness.length)
    : null;

  const avgElbow = Math.round(log.reduce((a, l) => a + l.elbowHeight, 0) / log.length * 100) / 100;

  const validTension = log.filter(l => l.shoulderTension > 0).map(l => l.shoulderTension);
  const avgTension = validTension.length > 0
    ? Math.round(validTension.reduce((a, b) => a + b, 0) / validTension.length)
    : 0;
  const maxTension = validTension.length > 0 ? Math.max(...validTension) : 0;

  // 弓配分
  const zoneCounts = { tip: 0, middle: 0, frog: 0 };
  log.forEach(l => zoneCounts[l.bowZone]++);
  const totalZone = zoneCounts.tip + zoneCounts.middle + zoneCounts.frog;
  const tipPct = Math.round(zoneCounts.tip / totalZone * 100);
  const frogPct = Math.round(zoneCounts.frog / totalZone * 100);
  const midPct = 100 - tipPct - frogPct;

  // ステータス判定
  const straightStatus = avgStraightness === null ? 'good'
    : avgStraightness >= 85 ? 'good'
    : avgStraightness >= 65 ? 'warn' : 'bad';

  const elbowStatus = avgElbow < -0.15 ? 'warn'
    : avgElbow > 0.5 ? 'bad'
    : avgElbow > 0.35 ? 'warn' : 'good';

  const shoulderStatus = maxTension > 25 ? 'bad'
    : avgTension > 15 ? 'warn' : 'good';

  // 弓配分ステータス
  const maxZone = Math.max(tipPct, midPct, frogPct);
  const minZone = Math.min(tipPct, midPct, frogPct);
  const distStatus = (maxZone - minZone) < 20 ? 'good'
    : (maxZone - minZone) < 35 ? 'warn' : 'bad';

  // 総合スコア（各100点満点を加重平均）
  const scores = [];
  if (avgStraightness !== null) scores.push({ s: avgStraightness, w: 3 });
  scores.push({ s: elbowStatus === 'good' ? 90 : elbowStatus === 'warn' ? 60 : 30, w: 2 });
  scores.push({ s: shoulderStatus === 'good' ? 95 : shoulderStatus === 'warn' ? 60 : 30, w: 2 });
  scores.push({ s: distStatus === 'good' ? 90 : distStatus === 'warn' ? 65 : 35, w: 1 });
  const totalW = scores.reduce((a, s) => a + s.w, 0);
  const overall = Math.round(scores.reduce((a, s) => a + s.s * s.w, 0) / totalW);
  const overallStatus = overall >= 80 ? 'good' : overall >= 60 ? 'warn' : 'bad';

  const overallComment = overall >= 85 ? '素晴らしいフォームです！この調子で練習を続けましょう。'
    : overall >= 70 ? '基本は良好です。下のポイントを意識するとさらに良くなります。'
    : overall >= 50 ? 'いくつか改善ポイントがあります。ゆっくり練習して修正しましょう。'
    : 'フォームに課題があります。鏡を見ながら1つずつ直していきましょう。';

  // 弓配分ラベル
  let distLabel = '全弓';
  if (maxZone >= 50) {
    if (tipPct === maxZone) distLabel = '先弓寄り';
    else if (frogPct === maxZone) distLabel = '元弓寄り';
    else distLabel = '中弓中心';
  }

  // 肘ラベル
  const elbowLabel = avgElbow < -0.15 ? '高すぎ'
    : avgElbow > 0.5 ? '低すぎ'
    : avgElbow > 0.35 ? 'やや低い' : '適正';

  // HTML生成
  let html = '';

  // 総合スコア
  html += `<div class="diagnose-overall">
    <div class="diagnose-overall-score" style="color:var(--${overallStatus})">${overall}点</div>
    <div class="diagnose-overall-label">総合スコア</div>
    <div class="diagnose-overall-comment">${overallComment}</div>
  </div>`;

  // 弓の直線性
  html += `<div class="diagnose-item ${straightStatus}">
    <div class="diagnose-item-label">弓の直線性</div>
    <div class="diagnose-item-value">${avgStraightness !== null ? avgStraightness + '%' : '検出不足'}</div>
    <div class="diagnose-item-detail">${
      straightStatus === 'good' ? '弓がまっすぐ引けています' :
      straightStatus === 'warn' ? 'やや曲がりが見られます。弦と直角を意識しましょう' :
      '弓が曲がっています。弓先の方向に注意してください'
    }</div>
  </div>`;

  // 弓の配分
  html += `<div class="diagnose-item ${distStatus}">
    <div class="diagnose-item-label">弓の配分</div>
    <div class="diagnose-item-value">${distLabel}</div>
    <div class="diagnose-item-detail">先弓${tipPct}% / 中弓${midPct}% / 元弓${frogPct}%</div>
  </div>`;

  // 肘の高さ
  html += `<div class="diagnose-item ${elbowStatus}">
    <div class="diagnose-item-label">肘の高さ</div>
    <div class="diagnose-item-value">${elbowLabel}</div>
    <div class="diagnose-item-detail">${
      elbowStatus === 'good' ? '肘の高さは適正です' :
      elbowLabel === '高すぎ' ? '肘が上がりすぎです。力まず自然な高さに' :
      elbowLabel === '低すぎ' ? '肘が下がりすぎです。弦の高さに合わせましょう' :
      '肘をもう少し上げてみましょう'
    }</div>
  </div>`;

  // 肩の緊張
  html += `<div class="diagnose-item ${shoulderStatus}">
    <div class="diagnose-item-label">肩の緊張</div>
    <div class="diagnose-item-value">${shoulderStatus === 'good' ? 'リラックス' : shoulderStatus === 'warn' ? '少し力み' : '力んでいます'}</div>
    <div class="diagnose-item-detail">平均${avgTension}% / 最大${maxTension}%${
      shoulderStatus !== 'good' ? '　息を吐いて肩を落としましょう' : ''
    }</div>
  </div>`;

  diagnoseResults.innerHTML = html;
  diagnoseModal.classList.remove('hidden');
}
