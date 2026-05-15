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
const scanActions = document.querySelector(".scan-actions");
const bicepsModeButton = document.querySelector("#mode-biceps");
const valgusModeButton = document.querySelector("#mode-valgus");
const cameraVideo = document.querySelector("#camera-video");
const cameraStatus = document.querySelector("#camera-status");
const bodyCanvas = document.querySelector("#body-canvas");
const canvasEmpty = document.querySelector("#canvas-empty");
const emptyGuidance = document.querySelector("#empty-guidance");
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
const scanTitle = document.querySelector("#scan-title");
const uploadGuidance = document.querySelector("#upload-guidance");
const captureProtocol = document.querySelector("#capture-protocol");
const stepDetect = document.querySelector("#step-detect");
const stepAngle = document.querySelector("#step-angle");
const stepMeasure = document.querySelector("#step-measure");
const stepResult = document.querySelector("#step-result");
const scanProgress = document.querySelector("#scan-progress");
const scanPercent = document.querySelector("#scan-percent");
const scanSteps = [...document.querySelectorAll(".scan-steps li")];
const canvasContext = bodyCanvas.getContext("2d");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const VALGUS_AUTO_SAVE_THRESHOLD = 94;
const VALGUS_CAMERA_WARMUP_MS = 15000;
const BICEPS_AUTO_SAVE_THRESHOLD = 94;
const BICEPS_MIN_ANGLE = 90;
const BICEPS_MAX_ANGLE = 91;

const POSE = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftPinky: 17,
  rightPinky: 18,
  leftIndex: 19,
  rightIndex: 20,
  leftThumb: 21,
  rightThumb: 22,
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
let latestValgusScan = null;
let savedValgusScan = null;
let valgusCameraStartedAt = 0;
let scanMode = "biceps";

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

function modeCopy() {
  if (scanMode === "valgus") {
    return {
      title: "Scan valgus du coude",
      empty: "Tendez les deux bras, paumes ouvertes face caméra.",
      upload: "Deux bras tendus, paumes ouvertes face caméra, coudes/poignets visibles.",
      summary: "Montrez les deux bras presque tendus face caméra pour relever le valgus gauche/droit.",
      live: "Détection live active. Tendez les deux bras, paumes ouvertes face caméra.",
      photo: "Photo importée. Analyse automatique du valgus avec paumes ouvertes visibles.",
      ready: "Modèle prêt. Tendez les deux bras, paumes ouvertes face caméra.",
      labels: ["Écart gauche/droite", "Qualité moyenne", "Profil valgus estimé"],
      steps: ["Deux bras détectés", "Coudes presque tendus", "Angles gauche/droit", "Relevé complété"],
      protocol: [
        "Paumes TOUJOURS ouvertes face caméra",
        "Deux bras tendus ou presque tendus",
        "Caméra bien en face, assez reculée",
      ],
    };
  }

  return {
    title: "Scan biceps",
    empty: "Levez un bras fléchi, coude et poignet bien visibles.",
    upload: "Bras levé et fléchi, poing fermé, épaule/coude/poignet visibles.",
    summary: "Levez un bras fléchi. L'app garde automatiquement le meilleur scan détecté.",
    live: "Détection live active. Levez un bras fléchi, épaule/coude/poignet visibles.",
    photo: "Photo importée. Analyse automatique du bras levé visible.",
    ready: "Modèle prêt. Levez un bras fléchi; le meilleur scan sera gardé automatiquement.",
    labels: ["Bras / avant-bras", "Qualité capture", "Profil biceps estimé"],
    steps: ["Épaule, coude, poignet", "Angle de flexion", "Bras / avant-bras", "Profil biceps"],
    protocol: [
      "Bras levé obligatoire",
      "Montrez surtout coude et poignet",
      "Épaule visible si possible, sans forcer",
    ],
  };
}

function applyModeCopy() {
  const copy = modeCopy();
  scanTitle.textContent = copy.title;
  emptyGuidance.textContent = copy.empty;
  uploadGuidance.textContent = copy.upload;
  primaryRatioLabel.textContent = copy.labels[0];
  secondaryRatioLabel.textContent = copy.labels[1];
  classificationOutput.previousElementSibling.textContent = copy.labels[2];
  stepDetect.textContent = copy.steps[0];
  stepAngle.textContent = copy.steps[1];
  stepMeasure.textContent = copy.steps[2];
  stepResult.textContent = copy.steps[3];
  captureProtocol.innerHTML = copy.protocol.map((item) => `<li>${item}</li>`).join("");
  scanActions.classList.toggle("compact", scanMode === "valgus");
  completeArmButton.textContent = "Valider ce bras";
  completeArmButton.hidden = true;
  useBestFrameButton.hidden = scanMode === "valgus";
  completeArmButton.disabled = true;
  useBestFrameButton.disabled = scanMode !== "biceps" || !bestArmScan;
  bicepsModeButton.classList.toggle("active", scanMode === "biceps");
  valgusModeButton.classList.toggle("active", scanMode === "valgus");
}

function resetResults(summary = modeCopy().summary) {
  const copy = modeCopy();
  primaryRatioLabel.textContent = copy.labels[0];
  secondaryRatioLabel.textContent = copy.labels[1];
  classificationOutput.previousElementSibling.textContent = copy.labels[2];
  limbOutput.textContent = "-";
  ratioPrimary.textContent = "-";
  ratioSecondary.textContent = "-";
  classificationOutput.textContent = "Non analysée";
  completeArmButton.disabled = true;
  useBestFrameButton.disabled = scanMode !== "biceps" || !bestArmScan;
  latestArmScan = null;
  latestValgusScan = null;
  savedValgusScan = null;
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

function drawHandPoints(points, color) {
  canvasContext.save();
  points.forEach((point) => {
    if ((point.visibility || 0) < 0.28) return;
    canvasContext.beginPath();
    canvasContext.arc(point.x, point.y, 5, 0, Math.PI * 2);
    canvasContext.fillStyle = color;
    canvasContext.fill();
    canvasContext.strokeStyle = "#ffffff";
    canvasContext.lineWidth = 2;
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

function valgusQuality(candidate) {
  const visibilityScore = candidate.reliability * 54;
  const extensionScore = Math.max(0, 1 - Math.abs(candidate.angle - 172) / 48) * 24;
  const sizeScore = Math.min(1, candidate.total / 125) * 17;
  const stabilityBonus = candidate.elbowReliable && candidate.wristReliable ? 5 : 0;
  return Math.round(Math.min(100, visibilityScore + extensionScore + sizeScore + stabilityBonus));
}

function armConfigs() {
  return [
    {
      type: "arm",
      side: "gauche",
      label: "Bras gauche",
      resultLabel: "Coude gauche",
      indices: [POSE.leftShoulder, POSE.leftElbow, POSE.leftWrist],
      handIndices: [POSE.leftPinky, POSE.leftIndex, POSE.leftThumb],
    },
    {
      type: "arm",
      side: "droit",
      label: "Bras droit",
      resultLabel: "Coude droit",
      indices: [POSE.rightShoulder, POSE.rightElbow, POSE.rightWrist],
      handIndices: [POSE.rightPinky, POSE.rightIndex, POSE.rightThumb],
    },
  ];
}

function chooseBestArm(landmarks, frame) {
  return armConfigs()
    .map((config) => armCandidate(landmarks, frame, config))
    .filter((candidate) => candidate.elbowReliable && candidate.wristReliable && candidate.total > 48)
    .sort((a, b) => captureQuality(b) - captureQuality(a))[0];
}

function chooseValgusArms(landmarks, frame) {
  return armConfigs()
    .map((config) => {
      const candidate = armCandidate(landmarks, frame, config);
      const hand = palmOpenSignal(landmarks, frame, config, candidate);
      return { ...candidate, hand };
    })
    .map((candidate) => ({
      ...candidate,
      quality: valgusQuality(candidate),
      measurable: candidate.elbowReliable && candidate.wristReliable && candidate.hand.open && candidate.total > 34,
    }));
}

function palmOpenSignal(landmarks, frame, config, candidate) {
  const handPoints = config.handIndices.map((index) => landmarkToCanvas(landmarks[index], frame));
  const visible = handPoints.filter((point) => point.visibility >= 0.28);
  if (visible.length < 3) {
    return { open: false, reason: "paume non visible", points: handPoints };
  }

  const [pinky, index, thumb] = handPoints;
  const palmWidth = Math.max(distance(pinky, index), distance(thumb, index), distance(thumb, pinky));
  const forearm = Math.max(candidate.forearm, 1);
  const spreadRatio = palmWidth / forearm;
  const awayFromWrist = handPoints.some((point) => distance(point, candidate.points[2]) / forearm > 0.12);
  const open = spreadRatio >= 0.14 && awayFromWrist;
  return {
    open,
    reason: open ? "paume ouverte" : "ouvrez la paume face caméra",
    points: handPoints,
  };
}

function classifyBiceps(candidate) {
  if (candidate.ratio >= 1.08) return "proxy biceps long";
  if (candidate.ratio <= 0.92) return "proxy biceps court";
  return "proxy biceps moyen";
}

function isArmRaised(candidate) {
  const [shoulder, elbow, wrist] = candidate.points;
  const segment = Math.max(candidate.total, 1);
  const wristAboveShoulder = wrist.y < shoulder.y - segment * 0.04;
  const elbowNearOrAboveShoulder = elbow.y < shoulder.y + segment * 0.08;
  return wristAboveShoulder && elbowNearOrAboveShoulder;
}

function isBicepsAngleValid(angle) {
  return angle >= BICEPS_MIN_ANGLE && angle <= BICEPS_MAX_ANGLE;
}

function valgusDeviation(candidate) {
  return Math.max(0, 180 - candidate.angle);
}

function classifyValgus(candidate) {
  const deviation = valgusDeviation(candidate);
  if (deviation <= 8) return "valgus faible / neutre";
  if (deviation <= 15) return "valgus modéré";
  return "valgus marqué";
}

function classifyValgusPair(results) {
  const complete = results.filter(Boolean);
  if (!complete.length) return "Non analysée";
  const marked = complete.filter((result) => result.deviation > 15).length;
  const moderate = complete.filter((result) => result.deviation > 8).length;
  if (marked) return "valgus marqué";
  if (moderate) return "valgus modéré";
  return "valgus faible / neutre";
}

function toArmScan(candidate) {
  return {
    side: candidate.side,
    label: candidate.label,
    ratio: candidate.ratio,
    quality: captureQuality(candidate),
    angle: candidate.angle,
    raised: isArmRaised(candidate),
    angleValid: isBicepsAngleValid(candidate.angle),
    classification: classifyBiceps(candidate),
    upperArm: candidate.upperArm,
    forearm: candidate.forearm,
  };
}

function bicepsGuidance(candidate) {
  const quality = captureQuality(candidate);
  if (!isArmRaised(candidate)) {
    return "Bras trop bas: levez le bras fléchi, avec le poignet au-dessus de l'épaule.";
  }
  if (!candidate.shoulderReliable) {
    return "Épaule faible: gardez le coude/poignet visibles et reculez légèrement pour inclure l'épaule.";
  }
  if (quality < 48) {
    return "Qualité faible: rapprochez le bras ou améliorez la lumière. Coude et poignet doivent rester visibles.";
  }
  if (!isBicepsAngleValid(candidate.angle)) {
    return `Angle incorrect: ajustez le coude entre ${BICEPS_MIN_ANGLE}° et ${BICEPS_MAX_ANGLE}°.`;
  }
  if (candidate.angle < 60 || candidate.angle > 145) {
    return "Angle difficile: fléchissez le bras pour rester entre environ 70° et 130°.";
  }
  return "Capture exploitable: estimation proxy basée sur les proportions bras/avant-bras.";
}

function valgusGuidance(candidate) {
  const quality = valgusQuality(candidate);
  if (!candidate.shoulderReliable || !candidate.elbowReliable || !candidate.wristReliable) {
    return "Repères incomplets: gardez épaule, coude et poignet dans le cadre.";
  }
  if (candidate.angle < 150) {
    return "Bras trop fléchi: tendez davantage le bras pour mesurer le valgus.";
  }
  if (quality < 52) {
    return "Qualité faible: placez le bras bien face caméra et améliorez la lumière.";
  }
  return "Capture exploitable: estimation 2D du valgus, à confirmer hors contexte médical.";
}

function toValgusResult(candidate) {
  const quality = valgusQuality(candidate);
  const deviation = valgusDeviation(candidate);
  return {
    side: candidate.side,
    label: candidate.resultLabel,
    deviation,
    angle: candidate.angle,
    quality,
    classification: classifyValgus(candidate),
    confidence: quality >= 62 ? "bonne confiance" : quality >= 45 ? "confiance correcte" : "confiance basse",
  };
}

function renderValgusCard(card, result, fallback) {
  if (!result) {
    card.innerHTML = fallback;
    card.classList.remove("complete");
    return;
  }
  card.classList.add("complete");
  card.innerHTML = `
    <span>${result.label}</span>
    <strong>${result.deviation.toFixed(1)}° · ${result.classification}</strong>
    <p>Angle ${Math.round(result.angle)}° · qualité ${result.quality}% · ${result.confidence}</p>
  `;
}

function renderValgusResults(results) {
  const left = results.find((result) => result?.side === "gauche");
  const right = results.find((result) => result?.side === "droit");
  renderValgusCard(
    firstArmCard,
    left,
    "<span>Coude gauche</span><strong>En attente</strong><p>Gardez coude et poignet gauches dans le cadre.</p>"
  );
  renderValgusCard(
    secondArmCard,
    right,
    "<span>Coude droit</span><strong>En attente</strong><p>Gardez coude et poignet droits dans le cadre.</p>"
  );

  if (!left && !right) {
    comparisonSummary.textContent = "Aucun relevé exploitable pour le moment. Reculez légèrement et montrez les deux bras.";
    return;
  }

  if (!left || !right) {
    const found = left || right;
    comparisonSummary.textContent = `${found.label} relevé (${found.deviation.toFixed(1)}°). Continuez: il manque l'autre coude pour compléter le relevé bilatéral.`;
    return;
  }

  const gap = Math.abs(left.deviation - right.deviation);
  const symmetry =
    gap <= 3
      ? "relevé symétrique"
      : gap <= 7
        ? "légère asymétrie"
        : "asymétrie marquée";
  comparisonSummary.textContent = `Relevé valgus complété: gauche ${left.deviation.toFixed(1)}°, droite ${right.deviation.toFixed(1)}°. ${symmetry}, écart ${gap.toFixed(1)}°.`;
}

function renderSavedValgusScan(scan) {
  renderValgusResults([scan.left, scan.right]);
  setProgress(100, 3);
  limbOutput.textContent = "Deux bras";
  ratioPrimary.textContent = `${scan.gap.toFixed(1)}°`;
  ratioSecondary.textContent = `${scan.quality}%`;
  classificationOutput.textContent = scan.classification;
  analysisConfidence.textContent = "Relevé enregistré";
  analysisSummary.textContent = `Relevé valgus enregistré automatiquement: gauche ${scan.left.deviation.toFixed(1)}°, droite ${scan.right.deviation.toFixed(1)}°, écart ${scan.gap.toFixed(1)}°.`;
  completeArmButton.disabled = true;
}

function renderModeResults() {
  if (scanMode === "valgus") {
    if (savedValgusScan) renderSavedValgusScan(savedValgusScan);
    else renderValgusResults([]);
  } else {
    renderCompletedScans();
  }
}

function renderValgusDetection(result) {
  if (savedValgusScan) {
    renderSavedValgusScan(savedValgusScan);
    return;
  }

  const frame = drawBaseFrame();
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) {
    setProgress(25, 0);
    analysisConfidence.textContent = "Aucune pose";
    analysisSummary.textContent = "Aucun bras détecté. Tendez les deux bras face caméra avec coudes et poignets visibles.";
    latestArmScan = null;
    latestValgusScan = null;
    completeArmButton.disabled = true;
    renderValgusResults([]);
    return;
  }

  const candidates = chooseValgusArms(landmarks, frame);
  const measurable = candidates.filter((candidate) => candidate.measurable);
  if (!measurable.length) {
    const armSeen = candidates.some((candidate) => candidate.elbowReliable && candidate.wristReliable && candidate.total > 34);
    const missingPalms = armSeen && candidates.some((candidate) => !candidate.hand.open);
    setProgress(45, 1);
    analysisConfidence.textContent = missingPalms ? "Paumes non validées" : "Bras incomplet";
    analysisSummary.textContent = missingPalms
      ? "Ouvrez les deux paumes face caméra. Le valgus ne sera pas enregistré sans paumes ouvertes visibles."
      : "Pose détectée, mais aucun coude/poignet n'est assez visible. Reculez ou écartez les bras.";
    latestArmScan = null;
    latestValgusScan = null;
    completeArmButton.disabled = true;
    renderValgusResults([]);
    return;
  }

  measurable.forEach((candidate) => {
    const color = candidate.side === "gauche" ? "#1f8a79" : "#cf5b43";
    drawLine(candidate.points, color);
    drawHandPoints(candidate.hand.points, color);
  });
  const results = measurable.map(toValgusResult);
  const left = results.find((scan) => scan.side === "gauche");
  const right = results.find((scan) => scan.side === "droit");
  const complete = Boolean(left && right);
  const quality = Math.round(results.reduce((sum, scan) => sum + scan.quality, 0) / results.length);
  const extensionReady = measurable.every((candidate) => candidate.angle >= 140);
  const progress = complete ? Math.max(72, quality) : Math.max(52, Math.round(quality * 0.8));
  const activeStep = complete ? 3 : extensionReady ? 2 : 1;
  const pairClassification = classifyValgusPair(results);
  const gap = complete ? Math.abs(left.deviation - right.deviation) : null;

  latestValgusScan = { left, right, complete, quality, classification: pairClassification, gap };
  setProgress(progress, activeStep);
  primaryRatioLabel.textContent = "Écart gauche/droite";
  secondaryRatioLabel.textContent = "Qualité moyenne";
  limbOutput.textContent = complete ? "Deux bras" : results[0].label;
  ratioPrimary.textContent = complete ? `${gap.toFixed(1)}°` : "partiel";
  ratioSecondary.textContent = `${quality}%`;
  classificationOutput.textContent = pairClassification;
  analysisConfidence.textContent = `${quality}% qualité`;
  completeArmButton.disabled = true;
  useBestFrameButton.disabled = true;
  latestArmScan = null;
  renderValgusResults(results);
  if (complete) {
    const warmupRemaining = Math.max(0, VALGUS_CAMERA_WARMUP_MS - (performance.now() - valgusCameraStartedAt));
    if (warmupRemaining > 0) {
      analysisSummary.textContent = `Calibration valgus: ${Math.ceil(warmupRemaining / 1000)}s restantes avant validation automatique. Gardez les bras tendus, paumes ouvertes face caméra.`;
      return;
    }
    if (quality >= VALGUS_AUTO_SAVE_THRESHOLD) {
      autoSaveValgusScan();
      return;
    }
    analysisSummary.textContent = `Relevé prêt, paumes ouvertes validées. Qualité ${quality}%; gardez les bras immobiles jusqu'à ${VALGUS_AUTO_SAVE_THRESHOLD}% pour l'enregistrement automatique.`;
  } else {
    analysisSummary.textContent = `${results[0].label} exploitable avec paume ouverte (${results[0].deviation.toFixed(1)}°). Ajoutez l'autre bras avec paume ouverte face caméra.`;
  }
}

function renderDetection(result) {
  if (scanMode === "valgus") {
    renderValgusDetection(result);
    return;
  }

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

  const raised = isArmRaised(candidate);
  const angleValid = isBicepsAngleValid(candidate.angle);
  const activeStep = quality > 55 && raised && angleValid ? 3 : !raised || !angleValid ? 1 : 2;
  setProgress(quality, activeStep);
  primaryRatioLabel.textContent = "Bras / avant-bras";
  secondaryRatioLabel.textContent = "Qualité capture";
  limbOutput.textContent = candidate.label;
  ratioPrimary.textContent = candidate.ratio.toFixed(2);
  ratioSecondary.textContent = `${quality}%`;
  classificationOutput.textContent = classification;
  analysisConfidence.textContent = `${quality}% qualité`;
  completeArmButton.disabled = true;

  if (!raised) {
    analysisSummary.textContent = `${buildGuidance(candidate)} L'enregistrement automatique est bloqué tant que le bras n'est pas levé.`;
    return;
  }

  if (!angleValid) {
    analysisSummary.textContent = `${buildGuidance(candidate)} L'enregistrement automatique est bloqué tant que l'angle n'est pas entre ${BICEPS_MIN_ANGLE}° et ${BICEPS_MAX_ANGLE}°.`;
    return;
  }

  if (canCompleteLatestScan()) {
    if (quality >= BICEPS_AUTO_SAVE_THRESHOLD) {
      autoSaveBicepsScan();
      return;
    }
    analysisSummary.textContent = `${buildGuidance(candidate)} Gardez le bras immobile jusqu'à ${BICEPS_AUTO_SAVE_THRESHOLD}% pour l'enregistrement automatique.`;
    return;
  }

  analysisSummary.textContent = buildGuidance(candidate);
}

function canCompleteLatestScan() {
  if (!latestArmScan || latestArmScan.quality < 50) return false;
  if (!latestArmScan.raised) return false;
  if (!latestArmScan.angleValid) return false;
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

function autoSaveValgusScan() {
  if (!latestValgusScan?.complete || savedValgusScan) return;
  savedValgusScan = {
    ...latestValgusScan,
    left: { ...latestValgusScan.left },
    right: { ...latestValgusScan.right },
    savedAt: new Date().toISOString(),
  };

  isCameraLive = false;
  cancelAnimationFrame(cameraFrameId);
  const snapshot = bodyCanvas.toDataURL("image/png");
  const image = new Image();
  image.onload = () => {
    analysisImage = image;
    drawBaseFrame();
    renderSavedValgusScan(savedValgusScan);
  };
  image.src = snapshot;
  cameraStatus.textContent = "Relevé valgus enregistré automatiquement.";
  renderSavedValgusScan(savedValgusScan);
}

function autoSaveBicepsScan() {
  if (!latestArmScan || !canCompleteLatestScan()) return;
  completedScans.push({ ...latestArmScan });
  bestArmScan = null;
  bestFrameDataUrl = "";
  useBestFrameButton.disabled = true;
  renderCompletedScans();

  if (completedScans.length === 1) {
    analysisSummary.textContent = `${latestArmScan.label} enregistré automatiquement. Montrez maintenant l'autre bras, même pose.`;
    latestArmScan = null;
    return;
  }

  isCameraLive = false;
  cancelAnimationFrame(cameraFrameId);
  const snapshot = bodyCanvas.toDataURL("image/png");
  const image = new Image();
  image.onload = () => {
    analysisImage = image;
    drawBaseFrame();
  };
  image.src = snapshot;
  cameraStatus.textContent = "Scan biceps enregistré automatiquement.";
  analysisSummary.textContent = "Scan biceps des deux bras enregistré automatiquement. Les résultats comparatifs sont affichés.";
  latestArmScan = null;
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
    analysisSummary.textContent = modeCopy().ready;
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
    if (scanMode === "valgus") {
      valgusCameraStartedAt = performance.now();
    }
    analysisImage = null;
    await setRunningMode("VIDEO");
    canvasEmpty.hidden = true;
    captureButton.disabled = false;
    runScanButton.disabled = !poseLandmarker;
    cameraStatus.textContent = "Caméra active.";
    resetResults(modeCopy().live);
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
    resetResults(modeCopy().photo);
    drawBaseFrame();
    detectCurrentFrame();
  };
  image.src = URL.createObjectURL(file);
}

function switchMode(mode) {
  if (scanMode === mode) return;
  scanMode = mode;
  if (isCameraLive) {
    if (scanMode === "valgus") valgusCameraStartedAt = performance.now();
  }
  applyModeCopy();
  renderModeResults();
  resetResults();
  if (analysisImage || isCameraLive) {
    detectCurrentFrame();
  }
}

function resetAnalysis() {
  isCameraLive = false;
  cancelAnimationFrame(cameraFrameId);
  lastResult = null;
  latestArmScan = null;
  bestArmScan = null;
  bestFrameDataUrl = "";
  latestValgusScan = null;
  savedValgusScan = null;
  valgusCameraStartedAt = 0;
  completedScans = [];
  renderModeResults();
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
bicepsModeButton.addEventListener("click", () => switchMode("biceps"));
valgusModeButton.addEventListener("click", () => switchMode("valgus"));

prepareSteps();
applyModeCopy();
renderModeResults();
resetResults("Chargement du modèle de détection biceps...");
initPoseModel();
