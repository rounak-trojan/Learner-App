// Add-Learner page: switches between the three enrollment sources (Upload File /
// Drive Link / Live Camera), and, for Live Camera, walks the person through a
// guided front / left / right head-turn scan, capturing frames along the way and
// stashing them as JSON in the #capture_frames hidden field for the surrounding
// <form> to submit like any other field.

(function initSourceTabs() {
  const tabs = document.querySelectorAll(".enroll-tab-btn");
  if (!tabs.length) return;

  const panels = {
    file: document.getElementById("panel-file"),
    drive: document.getElementById("panel-drive"),
    camera: document.getElementById("panel-camera"),
  };

  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.toggle("active", b === btn));
      Object.entries(panels).forEach(([key, el]) => {
        if (el) el.style.display = key === btn.dataset.source ? "block" : "none";
      });
      // Leaving the camera tab mid-scan stops the stream rather than leaving it
      // running in the background.
      if (btn.dataset.source !== "camera" && window.__faceRegStop) {
        window.__faceRegStop();
      }
    });
  });
})();

(function initFaceScanner() {
  const video = document.getElementById("reg-video");
  const canvas = document.getElementById("reg-canvas");
  const startBtn = document.getElementById("reg-start-btn");
  const retakeBtn = document.getElementById("reg-retake-btn");
  const stageLabel = document.getElementById("reg-stage-label");
  const instruction = document.getElementById("reg-instruction");
  const progressFill = document.getElementById("reg-progress-fill");
  const statusBox = document.getElementById("reg-status");
  const captureFileInput = document.getElementById("capture_frames_file");
  const submitBtn = document.getElementById("main-submit-btn");

  if (!video || !startBtn) return; // this page has no live-camera panel

  const STAGES = [
    { key: "front", label: "Face forward", instruction: "Look straight into the camera. Keep your head level." },
    { key: "left", label: "Turn left", instruction: "Slowly turn your head to your LEFT until your cheek shows." },
    { key: "right", label: "Turn right", instruction: "Slowly turn your head to your RIGHT until your cheek shows." },
  ];
  const FRAMES_PER_STAGE = 6;     // enough for solid pose diversity without a bloated payload
  const CAPTURE_WINDOW_MS = 2000; // frames for a stage are spread across this window
  const PREP_SECONDS = 3;         // on-screen countdown before each stage starts capturing
  const JPEG_QUALITY = 0.8;       // embeddings run on a 160x160 crop - full quality isn't needed

  let stream = null;
  let capturedFrames = [];
  let running = false;

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function setInstruction(label, text) {
    stageLabel.textContent = label;
    instruction.textContent = text;
  }

  function setProgress(pct) {
    progressFill.style.width = pct + "%";
  }

  async function startCamera() {
    // Registration doesn't need scanner-grade resolution - the embedder works off a
    // 160x160 crop regardless, so a smaller capture keeps the upload light.
    const attempts = [
      { video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" }, audio: false },
      { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    ];
    for (const constraints of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        return true;
      } catch (err) {
        /* try the next, lower-resolution constraint set */
      }
    }
    setInstruction("Camera unavailable", "Could not access the camera. Check permissions and try again.");
    return false;
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }
  window.__faceRegStop = stopCamera;

  function captureFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  }

  async function countdown(seconds, label, text) {
    for (let s = seconds; s > 0; s--) {
      setInstruction(label, `${text}  (starting in ${s}…)`);
      await sleep(1000);
    }
  }

  async function runStage(stage, stageIndex) {
    await countdown(PREP_SECONDS, stage.label, stage.instruction);
    setInstruction(stage.label, stage.instruction + "  Hold it…");
    const perFrameGap = CAPTURE_WINDOW_MS / FRAMES_PER_STAGE;
    for (let i = 0; i < FRAMES_PER_STAGE; i++) {
      capturedFrames.push(captureFrame());
      const overallDone = stageIndex * FRAMES_PER_STAGE + i + 1;
      setProgress(Math.round((overallDone / (STAGES.length * FRAMES_PER_STAGE)) * 100));
      await sleep(perFrameGap);
    }
  }

  async function runFullCapture() {
    if (running) return;
    running = true;
    capturedFrames = [];
    captureFileInput.value = "";
    if (submitBtn) submitBtn.disabled = true;
    startBtn.disabled = true;
    retakeBtn.style.display = "none";
    statusBox.textContent = "";
    setProgress(0);

    const camOk = await startCamera();
    if (!camOk) {
      running = false;
      startBtn.disabled = false;
      return;
    }
    await sleep(500); // let exposure/focus settle before the first capture

    for (let i = 0; i < STAGES.length; i++) {
      await runStage(STAGES[i], i);
    }

    stopCamera();

    // Sent as a real file part (not a form field) so it isn't subject to Werkzeug's
    // ~500KB cap on plain multipart text fields - a few MB of captured frames would
    // blow straight through that and the server would reject the whole request.
    const jsonBlob = new Blob([JSON.stringify(capturedFrames)], { type: "application/json" });
    const dt = new DataTransfer();
    dt.items.add(new File([jsonBlob], "capture_frames.json", { type: "application/json" }));
    captureFileInput.files = dt.files;

    setInstruction("Captured", `${capturedFrames.length} frames captured across front, left and right poses.`);
    setProgress(100);
    statusBox.textContent = `Face data captured (${capturedFrames.length} frames, ${(jsonBlob.size / 1024).toFixed(0)}KB). Submit the form, or retake.`;
    retakeBtn.style.display = "inline-block";
    startBtn.disabled = false;
    if (submitBtn) submitBtn.disabled = false;
    running = false;
  }

  startBtn.addEventListener("click", runFullCapture);
  retakeBtn.addEventListener("click", runFullCapture);
})();
