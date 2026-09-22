const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const resultPanel = document.getElementById("result-panel");
const resultIcon = document.getElementById("result-icon");
const resultText = document.getElementById("result-text");
const toggleBtn = document.getElementById("toggle-btn");
const recentLog = document.getElementById("recent-log");
const recentTitle = document.getElementById("recent-title");
const tabBtns = document.querySelectorAll(".tab-btn");

let mode = "login";               // "login" | "logout"
let scanning = false;             // starts paused — Start must be pressed
let busy = false;
const SCAN_INTERVAL_MS = 2500;

// Client-side cache of the last 10 punches per mode, so switching tabs is instant.
const recentCache = { login: [], logout: [] };

function setPanel(state, icon, text) {
  resultPanel.className = "result-panel result-" + state;
  resultIcon.textContent = icon;
  resultText.textContent = text;
}

function renderRecent() {
  recentTitle.textContent = mode === "login" ? "Last 10 — Logged In" : "Last 10 — Logged Out";
  recentLog.innerHTML = "";
  const items = recentCache[mode];
  if (!items.length) {
    const div = document.createElement("div");
    div.className = "log-line";
    div.textContent = "No punches yet.";
    recentLog.appendChild(div);
    return;
  }
  for (const item of items.slice(0, 10)) {
    const div = document.createElement("div");
    div.className = "log-line log-success";
    div.textContent = item.name + (item.roll_no ? " (Roll " + item.roll_no + ")" : "");
    recentLog.appendChild(div);
  }
}

function pushRecent(forMode, name, roll_no) {
  recentCache[forMode].unshift({ name, roll_no });
  recentCache[forMode] = recentCache[forMode].slice(0, 10);
  if (forMode === mode) renderRecent();
}

async function loadRecentFromServer(forMode) {
  try {
    const resp = await fetch(`/api/scan/recent?mode=${forMode}`);
    const data = await resp.json();
    recentCache[forMode] = (data.names || []).map((n) => ({ name: n.name, roll_no: n.roll_no }));
    if (forMode === mode) renderRecent();
  } catch (err) {
    // Non-fatal — the box just starts empty and fills in as scans happen.
  }
}

async function startCamera() {
  // Ask for the highest resolution the device offers, with a 1080p target. A wider frame
  // is what actually lets the server-side detector pick out a face 2m+ back, or pick out
  // 10-12 faces at once in a group shot - a 640x480 stream just doesn't have the pixels
  // for that regardless of how good the detection model is.
  const constraintAttempts = [
    { video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" }, audio: false },
    { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false },
    { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
  ];
  let lastErr = null;
  for (const constraints of constraintAttempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings ? track.getSettings() : {};
      setPanel("idle", "⏸", `Camera ready (${settings.width || "?"}×${settings.height || "?"}). Press Start to scan.`);
      return;
    } catch (err) {
      lastErr = err;
    }
  }
  setPanel("error", "⚠️", "Camera access denied or unavailable: " + (lastErr ? lastErr.message : "unknown error"));
}

function captureFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Higher JPEG quality than the previous 0.85 - group/far-field detection on the server
  // is sensitive to compression artifacts eating the fine detail it needs.
  return canvas.toDataURL("image/jpeg", 0.92);
}

async function runScan() {
  if (!scanning || busy) return;
  if (!video.videoWidth) return;
  busy = true;

  const imageData = captureFrame();
  try {
    const resp = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData, mode: mode }),
    });
    const data = await resp.json();
    handleResult(data);
  } catch (err) {
    setPanel("error", "⚠️", "Scan failed: " + err.message);
  } finally {
    busy = false;
  }
}

function handleResult(data) {
  if (data.status === "no_face") {
    setPanel("idle", "🔍", data.message || "No face detected");
    return;
  }
  if (data.status === "low_quality") {
    setPanel("idle", "🔍", data.message || "Hold steady");
    return;
  }
  if (data.status === "error") {
    setPanel("error", "⚠️", data.message || "Scan error");
    return;
  }
  if (data.status === "no_match") {
    setPanel("warn", "❓", data.message || "Face not recognized");
    return;
  }
  if (data.status === "ok") {
    // The server returns one entry per face it found in the frame - a class walking past
    // together produces multiple entries in a single scan, not just one. Show a summary
    // in the main panel and log every individual result underneath.
    const r = data.results[0];
    const markedCount = data.results.filter((x) => x.status === "marked").length;
    const notLoggedInCount = data.results.filter((x) => x.status === "not_logged_in").length;
    const modeLabel = mode === "login" ? "logged in" : "logged out";

    let summary;
    if (data.results.length > 1) {
      summary = `${data.results.length} faces detected — ${markedCount} ${modeLabel}`;
    } else if (r.status === "marked") {
      summary = `${r.name} (Roll ${r.roll_no}) — ${modeLabel}`;
    } else if (r.status === "already_marked") {
      summary = `${r.name} already ${modeLabel} today at ${r.marked_at}`;
    } else if (r.status === "not_logged_in") {
      summary = `${r.name} is not logged in — cannot log out`;
    } else {
      summary = `Face not recognized in this center's records`;
    }

    const icon = markedCount > 0 ? "✅" : notLoggedInCount > 0 ? "🚫" : r.status === "already_marked" ? "ℹ️" : "❓";
    const state = markedCount > 0 ? "success" : notLoggedInCount > 0 ? "error" : "warn";
    setPanel(state, icon, summary);

    for (const res of data.results) {
      if (res.status === "marked") {
        pushRecent(res.mode, res.name, res.roll_no);
      }
    }
  }
}

function switchMode(newMode) {
  mode = newMode;
  tabBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === newMode));
  renderRecent();
  setPanel("idle", scanning ? "🔍" : "⏸", scanning ? "Scanning…" : "Scanning paused");
}

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => switchMode(btn.dataset.mode));
});

toggleBtn.addEventListener("click", () => {
  scanning = !scanning;
  toggleBtn.textContent = scanning ? "Pause Scanning" : "Start Scanning";
  setPanel("idle", scanning ? "🔍" : "⏸", scanning ? "Scanning…" : "Scanning paused");
});

startCamera();
loadRecentFromServer("login");
loadRecentFromServer("logout");
renderRecent();
setInterval(runScan, SCAN_INTERVAL_MS);
