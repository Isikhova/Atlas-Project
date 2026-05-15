import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const photoInput = document.querySelector("#photo-input");
const startCameraButton = document.querySelector("#start-camera");
const captureButton = document.querySelector("#freeze-camera");
const runScanButton = document.querySelector("#run-scan");
const completeArmButton = document.querySelector("#complete-arm");
const useBestFrameButton = document.querySelector("#use-best-frame");
const resetAnalysisButton = document.querySelector("#reset-analysis");
const cameraVideo = document.querySelector("#camera-video");
const cameraStatus = document.querySelector("#camera-status");
const bodyCanvas = document.querySelector("#body-canvas");
const canvasEmpty = document.querySelector("#canvas-empty");
const analysisSummary = document.querySelector("#analysis-summary");
const analysisConfidence = document.querySelector("#analysis-confidence");
const primaryRatioLabel = document.querySelector("#primary-ratio-label");
const secondaryRatioLabel = document.querySelector("#secondary-ratio-label");
const limbOutput = document.querySelector("#limb-output");
const ratioPrimary = document.querySelector("#ratio-femur-tibia");
const ratioSecondary = document.querySelector("#ratio-femur-torso");
const classificationOutput = document.querySelector("#classification-output");
const firstArmCard = document.querySelector("#first-arm-card");
const secondArmCard = document.querySelector("#second-arm-card");
const comparisonSummary = document.querySelector("#comparison-summary");
const scanProgress = document.querySelector("#scan-progress");
const scanPercent = document.querySelector("#scan-percent");
const scanSteps = [...document.querySelectorAll(".scan-steps li")];
const canvasContext = bodyCanvas.getContext("2d");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const POSE = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
};

let poseLandmarker = null;
let analysisImage = null;
let cameraStream = null;
let isCameraLive = false;
let cameraFrameId = 0;
let lastVideoTime = -1;
let lastResult = null;
let runningMode = "VIDEO";
let latestArmScan = null;
let bestArmScan = null;
let bestFrameDataUrl = "";
let completedScans = [];

function prepareSteps() {
  scanSteps.forEach((step, index) => {
    step.dataset.index = String(index);
  });
}

function setProgress(percent, activeStep) {
  scanProgress.style.width = `${percent}%`;
  scanPercent.textContent = `${percent}%`;
  scanSteps.forEach((step) => {
    const index = Number(step.dataset.index);
    step.classList.toggle("active", index === activeStep);
    step.classList.toggle("done", index <= activeStep);
  });
}

function resetResults(summary = "Montrez un bras fléchi avec coude et poignet visibles, puis lancez l'analyse.") {
  primaryRatioLabel.textContent = "Bras / avant-bras";
  secondaryRatioLabel.textContent = "Qualité capture";
  limbOutput.textContent = "-";
  ratioPrimary.textContent = "-";
  ratioSecondary.textContent = "-";
  classificationOutput.textContent = "Non analysée";
  completeArmButton.disabled = true;
  useBestFrameButton.disabled = !bestArmScan;
  latestArmScan = null;
  analysisSummary.textContent = summary;
  if (poseLandmarker) analysisConfidence.textContent = "Prêt";
  setProgress(0, 0);
}

function setCanvasReady(summary) {
  canvasEmpty.hidden = true;
  runScanButton.disabled = !poseLandmarker;
  analysisSummary.textContent = summary;
}

function drawContain(source, sourceWidth, sourceHeight) {
  const scale = Math.min(bodyCanvas.width / sourceWidth, bodyCanvas.height / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  const x = (bodyCanvas.width - width) / 2;
  const y = (bodyCanvas.height - height) / 2;
  canvasContext.drawImage(source, x, y, width, height);
  return { x, y, width, height };
}

function drawBaseFrame() {
  canvasContext.clearRect(0, 0, bodyCanvas.width, bodyCanvas.height);
  canvasContext.fillStyle = "#101516";
  canvasContext.fillRect(0, 0, bodyCanvas.width, bodyCanvas.height);

  if (isCameraLive && cameraVideo.videoWidth) {
    return drawContain(cameraVideo, cameraVideo.videoWidth, cameraVideo.videoHeight);
  }
  if (analysisImage) {
    return drawContain(analysisImage, analysisImage.width, analysisImage.height);
  }
  return { x: 0, y: 0, width: bodyCanvas.width, height: bodyCanvas.height };
}

function landmarkToCanvas(landmark, frame) {
  return {
    x: frame.x + landmark.x * frame.width,
    y: frame.y + landmark.y * frame.height,
    visibility: landmark.visibility ?? landmark.presence ?? 1,
  };
}

function drawLine(points, color) {
  canvasContext.save();
  canvasContext.strokeStyle = color;
  canvasContext.lineWidth = 5;
  canvasContext.lineCap = "round";
  canvasContext.beginPath();
  points.forEach((point, index) => {
    if (index === 0) canvasContext.moveTo(point.x, point.y);
    else canvasContext.lineTo(point.x, point.y);
  });
  canvasContext.stroke();
  points.forEach((point) => {
    canvasContext.beginPath();
    canvasContext.arc(point.x, point.y, 7, 0, Math.PI * 2);
    canvasContext.fillStyle = "#ffffff";
    canvasContext.fill();
    canvasContext.strokeStyle = "#12483c";
    canvasContext.stroke();
  });
  canvasContext.restore();
}

function distance(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function confidenceFor(points) {
  return Math.min(...points.map((point) => point.visibility || 0));
}

function armCandidate(landmarks, frame, config) {
  const points = config.indices.map((index) => landmarkToCanvas(landmarks[index], frame));
  const visibility = confidenceFor(points);
  const upperArm = distance(points[0], points[1]);
  const forearm = distance(points[1], points[2]);
  const angle = elbowAngle(points[0], points[1], points[2]);
  const shoulderReliable = points[0].visibility >= 0.35;
  const elbowReliable = points[1].visibility >= 0.38;
  const wristReliable = points[2].visibility >= 0.38;
  const reliability = (points[0].visibility * 0.25) + (points[1].visibility * 0.4) + (points[2].visibility * 0.35);
  return {
    ...config,
    points,
    visibility,
    reliability,
    shoulderReliable,
    elbowReliable,
    wristReliable,
    upperArm,
    forearm,
    angle,
    ratio: upperArm / Math.max(forearm, 1),
    total: upperArm + forearm,
  };
}

function elbowAngle(shoulder, elbow, wrist) {
  const upper = { x: shoulder.x - elbow.x, y: shoulder.y - elbow.y };
  const lower = { x: wrist.x - elbow.x, y: wrist.y - elbow.y };
  const dot = upper.x * lower.x + upper.y * lower.y;
  const upperLength = Math.hypot(upper.x, upper.y);
  const lowerLength = Math.hypot(lower.x, lower.y);
  const cosine = dot / Math.max(upperLength * lowerLength, 1);
  return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
}

function captureQuality(candidate) {
  const visibilityScore = candidate.reliability * 48;
  const angleScore = Math.max(0, 1 - Math.abs(candidate.angle - 100) / 85) * 27;
  const sizeScore = Math.min(1, candidate.total / 170) * 20;
  const stabilityBonus = candidate.elbowReliable && candidate.wristReliable ? 5 : 0;
  return Math.round(Math.min(100, visibilityScore + angleScore + sizeScore + stabilityBonus));
}

function chooseBestArm(landmarks, frame) {
  const configs = [
    {
      type: "arm",
      side: "gauche",
      label: "Bras gauche",
      indices: [POSE.leftShoulder, POSE.leftElbow, POSE.leftWrist],
    },
    {
      type: "arm",
      side: "droit",
      label: "Bras droit",
      indices: [POSE.rightShoulder, POSE.rightElbow, POSE.rightWrist],
    },
  ];

  return configs
    .map((config) => armCandidate(landmarks, frame, config))
    .filter((candidate) => candidate.elbowReliable && candidate.wristReliable && candidate.total > 48)
    .sort((a, b) => captureQuality(b) - captureQuality(a))[0];
}

function classifyBiceps(candidate) {
  if (candidate.ratio >= 1.08) return "proxy biceps long";
  if (candidate.ratio <= 0.92) return "proxy biceps court";
  return "proxy biceps moyen";
}

function toArmScan(candidate) {
  return {
    side: candidate.side,
    label: candidate.label,
    ratio: candidate.ratio,
    quality: captureQuality(candidate),
    angle: candidate.angle,
    classification: classifyBiceps(candidate),
    upperArm: candidate.upperArm,
    forearm: candidate.forearm,
  };
}

function bicepsGuidance(candidate) {
  const quality = captureQuality(candidate);
  if (!candidate.shoulderReliable) {
    return "Épaule faible: gardez le coude/poignet visibles et reculez légèrement pour inclure l'épaule.";
  }
  if (quality < 48) {
    return "Qualité faible: rapprochez le bras ou améliorez la lumière. Coude et poignet doivent rester visibles.";
  }
  if (candidate.angle < 60 || candidate.angle > 145) {
    return "Angle difficile: fléchissez le bras pour rester entre environ 70° et 130°.";
  }
  return "Capture exploitable: estimation proxy basée sur les proportions bras/avant-bras.";
}

function renderDetection(result) {
  const frame = drawBaseFrame();
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) {
    setProgress(25, 0);
    analysisConfidence.textContent = "Aucune pose";
    analysisSummary.textContent = "Aucun bras détecté. Montrez un bras entier avec épaule, coude et poignet visibles.";
    latestArmScan = null;
    completeArmButton.disabled = true;
    return;
  }

  const candidate = chooseBestArm(landmarks, frame);
  if (!candidate) {
    setProgress(45, 1);
    analysisConfidence.textContent = "Bras incomplet";
    analysisSummary.textContent = "Pose détectée, mais épaule, coude ou poignet ne sont pas assez visibles.";
    latestArmScan = null;
    completeArmButton.disabled = true;
    return;
  }

  drawLine(candidate.points, "#cf5b43");
  const classification = classifyBiceps(candidate);
  const quality = captureQuality(candidate);
  latestArmScan = toArmScan(candidate);
  rememberBestScan(latestArmScan);

  const activeStep = quality > 55 ? 3 : candidate.angle < 60 || candidate.angle > 145 ? 1 : 2;
  setProgress(quality, activeStep);
  primaryRatioLabel.textContent = "Bras / avant-bras";
  secondaryRatioLabel.textContent = "Qualité capture";
  limbOutput.textContent = candidate.label;
  ratioPrimary.textContent = candidate.ratio.toFixed(2);
  ratioSecondary.textContent = `${quality}%`;
  classificationOutput.textContent = classification;
  analysisConfidence.textContent = `${quality}% qualité`;
  completeArmButton.disabled = !canCompleteLatestScan();
  analysisSummary.textContent = buildGuidance(candidate);
}

function canCompleteLatestScan() {
  if (!latestArmScan || latestArmScan.quality < 50) return false;
  if (completedScans.length === 0) return true;
  return latestArmScan.side !== completedScans[0].side;
}

function buildGuidance(candidate) {
  const scan = toArmScan(candidate);
  const base = `${bicepsGuidance(candidate)} Angle coude ${Math.round(candidate.angle)}°, ratio ${candidate.ratio.toFixed(2)}.`;
  if (scan.quality < 50) {
    return `${base} Qualité encore trop faible. Essayez de garder coude et poignet visibles, ou utilisez “Meilleur scan”.`;
  }
  if (completedScans.length === 0) {
    return `${base} Vous pouvez valider ce premier bras.`;
  }
  if (scan.side === completedScans[0].side) {
    return `${base} Premier bras déjà scanné: montrez l'autre bras pour compléter le scan.`;
  }
  return `${base} Vous pouvez valider le deuxième bras.`;
}

function rememberBestScan(scan) {
  if (!scan) return;
  const isDifferentSecondArm = completedScans.length === 0 || scan.side !== completedScans[0].side;
  if (!isDifferentSecondArm) return;
  if (!bestArmScan || scan.quality > bestArmScan.quality) {
    bestArmScan = { ...scan };
    bestFrameDataUrl = bodyCanvas.toDataURL("image/png");
    useBestFrameButton.disabled = false;
  }
}

function useBestFrame() {
  if (!bestArmScan) return;
  latestArmScan = { ...bestArmScan };
  limbOutput.textContent = latestArmScan.label;
  ratioPrimary.textContent = latestArmScan.ratio.toFixed(2);
  ratioSecondary.textContent = `${latestArmScan.quality}%`;
  classificationOutput.textContent = latestArmScan.classification;
  analysisConfidence.textContent = `${latestArmScan.quality}% qualité`;
  completeArmButton.disabled = !canCompleteLatestScan();
  analysisSummary.textContent = `Meilleur scan repris (${latestArmScan.label}, qualité ${latestArmScan.quality}%). Vous pouvez le valider si le bras affiché correspond bien.`;

  if (bestFrameDataUrl) {
    const image = new Image();
    image.onload = () => {
      analysisImage = image;
      isCameraLive = false;
      cancelAnimationFrame(cameraFrameId);
      drawBaseFrame();
    };
    image.src = bestFrameDataUrl;
  }
}

function renderArmCard(card, scan, fallback) {
  if (!scan) {
    card.innerHTML = fallback;
    card.classList.remove("complete");
    return;
  }
  card.classList.add("complete");
  card.innerHTML = `
    <span>${scan.label}</span>
    <strong>${scan.classification}</strong>
    <p>Ratio ${scan.ratio.toFixed(2)} · qualité ${scan.quality}% · angle ${Math.round(scan.angle)}°</p>
  `;
}

function renderCompletedScans() {
  renderArmCard(
    firstArmCard,
    completedScans[0],
    "<span>Bras 1</span><strong>En attente</strong><p>Validez un premier bras quand la qualité est suffisante.</p>"
  );
  renderArmCard(
    secondArmCard,
    completedScans[1],
    "<span>Bras 2</span><strong>En attente</strong><p>L'app vous guidera vers le deuxième bras après validation du premier.</p>"
  );

  if (completedScans.length < 2) {
    comparisonSummary.textContent =
      completedScans.length === 1
        ? `Premier scan enregistré (${completedScans[0].label}). Montrez l'autre bras pour compléter la comparaison.`
        : "Complétez les deux bras pour afficher la comparaison.";
    return;
  }

  const [first, second] = completedScans;
  const ratioGap = Math.abs(first.ratio - second.ratio);
  const qualityAverage = Math.round((first.quality + second.quality) / 2);
  const symmetry =
    ratioGap < 0.05
      ? "proportions très proches"
      : ratioGap < 0.1
        ? "légère différence de proportion"
        : "différence notable de proportion";

  comparisonSummary.textContent = `Scan complet: ${first.label} et ${second.label}. ${symmetry}; écart ratio ${ratioGap.toFixed(2)}, qualité moyenne ${qualityAverage}%.`;
}

function completeCurrentArm() {
  if (!latestArmScan || !canCompleteLatestScan()) return;
  completedScans.push(latestArmScan);
  bestArmScan = null;
  bestFrameDataUrl = "";
  useBestFrameButton.disabled = true;
  renderCompletedScans();

  if (completedScans.length === 1) {
    analysisSummary.textContent = `${latestArmScan.label} enregistré. Montrez maintenant l'autre bras, même pose, puis validez quand la qualité est suffisante.`;
    completeArmButton.disabled = true;
    return;
  }

  analysisSummary.textContent = "Scan des deux bras terminé. Les résultats comparatifs sont affichés.";
  completeArmButton.disabled = true;
}

async function initPoseModel() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    analysisConfidence.textContent = "Prêt";
    analysisSummary.textContent = "Modèle prêt. Montrez un bras fléchi; le meilleur scan sera gardé automatiquement.";
    runScanButton.disabled = !analysisImage && !isCameraLive;
  } catch (error) {
    console.error("Pose model failed", error);
    analysisConfidence.textContent = "Modèle indisponible";
    analysisSummary.textContent = "Le modèle de détection n'a pas pu être chargé. Vérifiez la connexion réseau.";
  }
}

async function setRunningMode(mode) {
  if (runningMode === mode || !poseLandmarker) return;
  await poseLandmarker.setOptions({ runningMode: mode });
  runningMode = mode;
}

async function detectCurrentFrame() {
  if (!poseLandmarker) return;
  let result = null;
  if (isCameraLive && cameraVideo.videoWidth) {
    await setRunningMode("VIDEO");
    result = poseLandmarker.detectForVideo(cameraVideo, performance.now());
  } else if (analysisImage) {
    drawBaseFrame();
    await setRunningMode("IMAGE");
    result = poseLandmarker.detect(bodyCanvas);
  }
  lastResult = result;
  renderDetection(result);
}

function drawCameraLoop() {
  if (!isCameraLive) return;
  if (cameraVideo.currentTime !== lastVideoTime) {
    lastVideoTime = cameraVideo.currentTime;
    if (poseLandmarker) {
      if (runningMode !== "VIDEO") {
        poseLandmarker.setOptions({ runningMode: "VIDEO" });
        runningMode = "VIDEO";
      }
      const result = poseLandmarker.detectForVideo(cameraVideo, performance.now());
      lastResult = result;
      renderDetection(result);
    } else {
      drawBaseFrame();
    }
  }
  cameraFrameId = requestAnimationFrame(drawCameraLoop);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraStatus.textContent = "Caméra non disponible dans ce navigateur.";
    return;
  }

  try {
    cameraStatus.textContent = "Demande d'autorisation caméra...";
    const preferredConstraints = {
      video: {
        facingMode: "user",
        width: { ideal: 960 },
        height: { ideal: 1280 },
      },
      audio: false,
    };
    const fallbackConstraints = { video: true, audio: false };
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia(preferredConstraints);
    } catch (preferredError) {
      cameraStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
    }
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    isCameraLive = true;
    analysisImage = null;
    await setRunningMode("VIDEO");
    canvasEmpty.hidden = true;
    captureButton.disabled = false;
    runScanButton.disabled = !poseLandmarker;
    cameraStatus.textContent = "Caméra active.";
    resetResults("Détection live active. Montrez un bras fléchi, épaule/coude/poignet visibles.");
    drawCameraLoop();
  } catch (error) {
    const errorName = error?.name || "Erreur inconnue";
    const advice = {
      NotAllowedError: "Autorisez la caméra dans le navigateur et dans les réglages système.",
      NotFoundError: "Aucune caméra détectée.",
      NotReadableError: "La caméra est peut-être utilisée par une autre application.",
      OverconstrainedError: "La caméra ne correspond pas aux contraintes demandées.",
      SecurityError: "L'accès caméra exige localhost ou HTTPS.",
    };
    console.error("Camera access failed", error);
    cameraStatus.textContent = `Caméra indisponible (${errorName}). ${advice[errorName] || "Essayez Chrome/Safari."}`;
  }
}

function captureCameraFrame() {
  if (!cameraVideo.videoWidth) return;
  isCameraLive = false;
  cancelAnimationFrame(cameraFrameId);
  const image = new Image();
  drawBaseFrame();
  image.onload = () => {
    analysisImage = image;
    cameraStatus.textContent = "Image capturée.";
    runScanButton.disabled = !poseLandmarker;
    detectCurrentFrame();
  };
  image.src = bodyCanvas.toDataURL("image/png");
}

function loadAnalysisImage(file) {
  const image = new Image();
  image.onload = () => {
    analysisImage = image;
    isCameraLive = false;
    cancelAnimationFrame(cameraFrameId);
    canvasEmpty.hidden = true;
    cameraStatus.textContent = "Photo importée.";
    runScanButton.disabled = !poseLandmarker;
    resetResults("Photo importée. Analyse automatique du bras visible.");
    drawBaseFrame();
    detectCurrentFrame();
  };
  image.src = URL.createObjectURL(file);
}

function resetAnalysis() {
  isCameraLive = false;
  cancelAnimationFrame(cameraFrameId);
  lastResult = null;
  latestArmScan = null;
  bestArmScan = null;
  bestFrameDataUrl = "";
  completedScans = [];
  renderCompletedScans();
  resetResults();
  if (analysisImage) {
    drawBaseFrame();
    runScanButton.disabled = !poseLandmarker;
  } else {
    canvasEmpty.hidden = false;
    runScanButton.disabled = true;
    captureButton.disabled = !cameraStream;
    drawBaseFrame();
  }
}

photoInput.addEventListener("change", () => {
  const [file] = [...photoInput.files];
  if (file) loadAnalysisImage(file);
});

startCameraButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", captureCameraFrame);
runScanButton.addEventListener("click", detectCurrentFrame);
completeArmButton.addEventListener("click", completeCurrentArm);
useBestFrameButton.addEventListener("click", useBestFrame);
resetAnalysisButton.addEventListener("click", resetAnalysis);

prepareSteps();
renderCompletedScans();
resetResults("Chargement du modèle de détection biceps...");
initPoseModel();
