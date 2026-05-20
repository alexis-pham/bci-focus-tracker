/**
 * content.js — Injects a distraction overlay banner into the active page.
 *
 * Listens for BCI_STATE messages from popup.js. Shows a red banner at the
 * top of the screen when distracted, removes it when focused.
 *
 * The overlay is non-blocking — it sits above page content but does not
 * prevent interaction with the page.
 */

(function () {
  const OVERLAY_ID = "bci-focus-overlay";

  // ── Create overlay element ──────────────────────────────────────────────
  function createOverlay() {
    const el = document.createElement("div");
    el.id = OVERLAY_ID;

    Object.assign(el.style, {
      position:        "fixed",
      top:             "0",
      left:            "0",
      right:           "0",
      zIndex:          "2147483647",   // max z-index
      padding:         "10px 20px",
      background:      "linear-gradient(90deg, #c0392b, #e74c3c)",
      color:           "#fff",
      fontFamily:      "'SF Mono', 'Fira Code', 'Consolas', monospace",
      fontSize:        "13px",
      fontWeight:      "600",
      letterSpacing:   "0.06em",
      display:         "flex",
      alignItems:      "center",
      justifyContent:  "space-between",
      boxShadow:       "0 2px 16px rgba(231, 76, 60, 0.5)",
      transform:       "translateY(-100%)",
      transition:      "transform 0.35s cubic-bezier(0.4, 0, 0.2, 1)",
      pointerEvents:   "none",   // don't block clicks
    });

    el.innerHTML = `
      <span>
        🧠 &nbsp;
        <strong>DISTRACTION DETECTED</strong>
        &nbsp;— EEG signal indicates loss of focus
      </span>
      <span style="opacity:0.7;font-size:11px;letter-spacing:0.04em">
        BCI Focus Monitor
      </span>
    `;

    document.body.appendChild(el);
    return el;
  }

  function getOrCreateOverlay() {
    return document.getElementById(OVERLAY_ID) || createOverlay();
  }

  // ── Show / hide ───────────────────────────────────────────────────────
  function showOverlay() {
    const el = getOrCreateOverlay();
    // Small delay lets the browser register the initial transform before
    // transitioning — prevents the animation from being skipped
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.style.transform = "translateY(0)";
      });
    });
  }

  function hideOverlay() {
    const el = document.getElementById(OVERLAY_ID);
    if (el) el.style.transform = "translateY(-100%)";
  }

  // ── Listen for messages from popup.js ────────────────────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type !== "BCI_STATE") return;

    if (msg.prediction === "distracted") {
      showOverlay();
    } else {
      hideOverlay();
    }
  });
})();
