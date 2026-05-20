/**
 * popup.js — Polls the replay server and updates the popup UI.
 *
 * Polls /status every second. Sends the current prediction to the
 * content script via chrome.tabs so the page overlay stays in sync.
 */

const SERVER = "http://localhost:5000";
const POLL_MS = 1000;

const stateCard    = document.getElementById("stateCard");
const dot          = document.getElementById("dot");
const stateLabel   = document.getElementById("stateLabel");
const statWindow   = document.getElementById("statWindow");
const statElapsed  = document.getElementById("statElapsed");
const progressFill = document.getElementById("progressFill");
const connDot      = document.getElementById("connDot");
const connLabel    = document.getElementById("connLabel");
const sessionMsg   = document.getElementById("sessionMsg");

// ── Messages shown for each state ────────────────────────────────────────
const MESSAGES = {
  focused:    "✓ Neural activity consistent with focused state.",
  distracted: "⚠ Distraction detected — try to refocus.",
  artifact:   "~ Signal artifact — window skipped.",
  waiting:    "Waiting for replay to start...",
  done:       "Replay session complete.",
  unknown:    "Connecting...",
};

let connected = false;

function setConnected(ok) {
  connected = ok;
  connDot.className   = `conn-dot ${ok ? "connected" : "disconnected"}`;
  connLabel.textContent = ok
    ? "Connected to replay server"
    : "Cannot reach replay server";
}

function applyState(prediction) {
  // Card class
  stateCard.className = `state-card ${prediction}`;
  dot.className       = `dot ${prediction}`;
  stateLabel.className = `state-label ${prediction}`;

  // Label text
  const labels = {
    focused:    "FOCUSED",
    distracted: "DISTRACTED",
    artifact:   "ARTIFACT",
    waiting:    "WAITING",
    done:       "DONE",
    unknown:    "—",
  };
  stateLabel.textContent = labels[prediction] ?? prediction.toUpperCase();

  // Message
  sessionMsg.textContent = MESSAGES[prediction] ?? "";
  sessionMsg.className   = `session-msg ${
    prediction === "distracted" ? "distracted-msg" :
    prediction === "focused"    ? "focused-msg"    : ""
  }`;
}

async function poll() {
  try {
    const res  = await fetch(`${SERVER}/status`, { signal: AbortSignal.timeout(900) });
    const data = await res.json();

    setConnected(true);
    applyState(data.prediction);

    // Stats
    const w = data.window_idx ?? 0;
    const n = data.total_windows ?? 0;
    statWindow.textContent  = n > 0 ? `${w + 1} / ${n}` : "—";
    statElapsed.textContent = data.elapsed_sec != null
      ? `${data.elapsed_sec.toFixed(1)}s`
      : "—";

    // Progress bar
    progressFill.style.width = `${data.progress_pct ?? 0}%`;

    // Tell content script to show / hide overlay
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          type:       "BCI_STATE",
          prediction: data.prediction,
        }).catch(() => {});  // tab may not have content script — ignore
      }
    });

  } catch {
    setConnected(false);
    applyState("unknown");
  }
}

// Start polling immediately, then every POLL_MS
poll();
setInterval(poll, POLL_MS);
