/**
 * background.js — Minimal service worker for Manifest V3.
 *
 * No persistent logic needed here — the popup polls the server directly
 * and sends messages to content scripts itself. This file exists only
 * because MV3 requires a service worker to be declared.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.log("BCI Focus Monitor installed.");
});
