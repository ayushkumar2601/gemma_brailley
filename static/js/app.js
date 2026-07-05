/**
 * app.js — Braille Vision frontend
 * Camera scanning, image upload, TTS, history, confidence meter, theme.
 */
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let isPaused            = false;
let lastText            = '';
let historyCount        = 0;
let frameErrors         = 0;
const MAX_ERRORS        = 10;
let browserCameraActive = false;
let browserStream       = null;
let browserLoopId       = null;
let frameInFlight       = false;
let lastSendTime        = 0;
const SEND_INTERVAL_MS  = 300;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const camImg         = document.getElementById('camera-img');
const noCamera       = document.getElementById('no-camera');
const scanOverlay    = document.getElementById('scan-overlay');
const uploadOverlay  = document.getElementById('upload-overlay');
const statusBadge    = document.getElementById('status-badge');
const statsLabel     = document.getElementById('stats-label');
const lastTextEl     = document.getElementById('last-text');
const aiBadge        = document.getElementById('ai-badge');
const historyList    = document.getElementById('history-list');
const historyCountEl = document.getElementById('history-count');
const qQuality       = document.getElementById('q-quality');
const qDots          = document.getElementById('q-dots');
const qFrames        = document.getElementById('q-frames');
const qBrailleFrames = document.getElementById('q-braille-frames');
const qElapsed       = document.getElementById('q-elapsed');
const btnPause       = document.getElementById('btn-pause');
const btnCamera      = document.getElementById('btn-camera');
const webcamVideo    = document.getElementById('webcam-video');
const webcamCanvas   = document.getElementById('webcam-canvas');
const toast          = document.getElementById('toast');
const confFill       = document.getElementById('conf-fill');
const confPct        = document.getElementById('conf-pct');
const indGroq        = document.getElementById('ind-groq');
const indCnn         = document.getElementById('ind-cnn');
const voiceSelect    = document.getElementById('voice-select');
const cameraWrap     = document.getElementById('camera-wrap');

// ── Theme ─────────────────────────────────────────────────────────────────────
function toggleTheme() {
  document.body.classList.toggle('light');
  const isLight = document.body.classList.contains('light');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
  showToast(isLight ? '☀ Light mode' : '🌙 Dark mode');
}
(function initTheme() {
  if (localStorage.getItem('theme') === 'light') document.body.classList.add('light');
})();

// ── Confidence meter ──────────────────────────────────────────────────────────
function setConfidence(value) {
  const pct = Math.round(value * 100);
  confFill.style.width = pct + '%';
  confPct.textContent  = pct > 0 ? pct + '%' : '—';
  confFill.className   = 'conf-fill ' + (pct < 40 ? 'low' : pct < 70 ? 'medium' : 'high');
}

// ── Live preview from webcam ──────────────────────────────────────────────────
function showLivePreview() {
  if (!browserCameraActive || !webcamVideo.videoWidth) return;
  const w = webcamVideo.videoWidth, h = webcamVideo.videoHeight;
  const maxW = 640, scale = w > maxW ? maxW / w : 1;
  webcamCanvas.width  = Math.round(w * scale);
  webcamCanvas.height = Math.round(h * scale);
  webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  camImg.src = webcamCanvas.toDataURL('image/jpeg', 0.6);
  camImg.style.display = 'block';
  noCamera.style.display = 'none';
}

// ── Poll server frame (OpenCV fallback) ───────────────────────────────────────
async function pollFrame() {
  if (!browserCameraActive) {
    try {
      const res  = await fetch('/api/frame');
      const data = await res.json();
      if (data.frame) {
        camImg.src = 'data:image/jpeg;base64,' + data.frame;
        camImg.style.display = 'block';
        noCamera.style.display = 'none';
        frameErrors = 0;
      } else { frameErrors++; }
    } catch (e) { frameErrors++; }
    if (frameErrors > MAX_ERRORS) {
      camImg.style.display = 'none';
      noCamera.style.display = 'flex';
    }
  }
  setTimeout(pollFrame, 250);
}

// ── Browser camera ────────────────────────────────────────────────────────────
async function startBrowserCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast('⚠ Browser camera not supported — try Chrome or Safari');
    return;
  }
  showToast('Requesting camera permission…');
  try {
    if (browserStream) browserStream.getTracks().forEach(t => t.stop());
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (_) {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    browserStream = stream;
    webcamVideo.srcObject = stream;
    await webcamVideo.play();
    browserCameraActive = true;
    frameErrors = 0;
    btnCamera.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Camera On`;
    btnCamera.disabled = true;
    btnCamera.classList.remove('btn-primary');
    btnCamera.classList.add('btn-secondary');
    if (scanOverlay) scanOverlay.style.display = 'block';

    function loop() {
      showLivePreview();
      const now = Date.now();
      if (!frameInFlight && !isPaused && now - lastSendTime >= SEND_INTERVAL_MS) sendBrowserFrame();
      browserLoopId = requestAnimationFrame(loop);
    }
    if (browserLoopId) cancelAnimationFrame(browserLoopId);
    browserLoopId = requestAnimationFrame(loop);
    showToast('📷 Camera live — scanning for Braille');
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showToast('❌ Camera blocked — use Upload Image instead');
      noCamera.querySelector('p').innerHTML =
        '<strong>Camera permission denied.</strong><br>Use <strong>Upload</strong> instead, or allow camera in browser settings.';
    } else if (err.name === 'NotFoundError') {
      showToast('❌ No camera found — use Upload Image');
    } else {
      showToast('❌ Camera error: ' + err.message);
    }
  }
}

// ── Send frame to server ──────────────────────────────────────────────────────
async function sendBrowserFrame() {
  if (!browserCameraActive || isPaused || frameInFlight) return;
  const w = webcamVideo.videoWidth, h = webcamVideo.videoHeight;
  if (!w || !h) return;
  const maxW = 480, scale = w > maxW ? maxW / w : 1;
  webcamCanvas.width  = Math.round(w * scale);
  webcamCanvas.height = Math.round(h * scale);
  webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  const dataUrl = webcamCanvas.toDataURL('image/jpeg', 0.55);
  frameInFlight = true;
  lastSendTime  = Date.now();
  try {
    const res  = await fetch('/api/process_frame', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: dataUrl }),
    });
    const data = await res.json();
    if (data.ok && data.frame) camImg.src = 'data:image/jpeg;base64,' + data.frame;
    if (data.confidence != null) setConfidence(data.confidence);
  } catch (e) { /* keep live preview */ }
  finally { frameInFlight = false; }
}

// ── Image upload ──────────────────────────────────────────────────────────────
function triggerUpload() { document.getElementById('file-input').click(); }

async function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = '';

  const reader = new FileReader();
  reader.onload = e => {
    camImg.src = e.target.result;
    camImg.style.display = 'block';
    noCamera.style.display = 'none';
  };
  reader.readAsDataURL(file);

  uploadOverlay.style.display = 'flex';
  showToast('Analysing Braille…');

  try {
    const formData = new FormData();
    formData.append('image', file);
    const res  = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    uploadOverlay.style.display = 'none';

    if (data.ok) {
      const displayText = data.corrected || data.text || '';
      if (displayText) {
        lastText = displayText;
        lastTextEl.textContent = displayText;
        lastTextEl.classList.add('flash');
        setTimeout(() => lastTextEl.classList.remove('flash'), 900);
        aiBadge.style.display = data.groq_used ? 'inline-flex' : 'none';
        setConfidence(data.confidence || 0);
        showToast(`✅ Detected: ${displayText.slice(0, 50)}`);
        speakInBrowser(displayText);
      } else {
        showToast('⚠ No Braille detected in image');
      }
      if (data.frame) camImg.src = 'data:image/jpeg;base64,' + data.frame;
      pollHistory();
    } else {
      showToast('❌ ' + (data.error || data.message || 'Processing failed'));
    }
  } catch (e) {
    uploadOverlay.style.display = 'none';
    showToast('❌ Upload failed: ' + e.message);
  }
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
cameraWrap.addEventListener('dragover', e => { e.preventDefault(); cameraWrap.classList.add('drag-over'); });
cameraWrap.addEventListener('dragleave', () => cameraWrap.classList.remove('drag-over'));
cameraWrap.addEventListener('drop', e => {
  e.preventDefault(); cameraWrap.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleUpload({ target: { files: [file], value: '' } });
});

// ── Status polling ────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    updateStatus(data);
  } catch (e) { /* ignore */ }
  setTimeout(pollStatus, 500);
}

async function pollHistory() {
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    updateHistory(data);
  } catch (e) { /* ignore */ }
  setTimeout(pollHistory, 1000);
}

function updateStatus(data) {
  const quality = data.quality || 'UNKNOWN';
  const stats   = data.stats   || {};

  statusBadge.textContent = qualityLabel(quality);
  statusBadge.className   = 'badge ' + qualityBadgeClass(quality, data.paused);

  let line = `${stats.elapsed_s ?? 0}s · ${stats.frames ?? 0} frames · ${stats.braille_frames ?? 0} braille detections`;
  if (data.confidence != null && data.confidence > 0)
    line += ` · ${(data.confidence * 100).toFixed(0)}% confidence`;
  if (statsLabel) statsLabel.textContent = line;

  qQuality.textContent       = qualityLabel(quality);
  qQuality.style.color       = qualityColour(quality);
  qDots.textContent          = data.dot_count ?? 0;
  qFrames.textContent        = stats.frames ?? 0;
  qBrailleFrames.textContent = stats.braille_frames ?? 0;
  if (qElapsed) qElapsed.textContent = (stats.elapsed_s ?? 0) + 's';

  if (data.confidence != null) setConfidence(data.confidence);

  if (data.groq_ok) indGroq.classList.add('active', 'ind-active');
  if (data.cnn_ok)  indCnn.classList.add('active', 'ind-active');

  const newText = data.last_text || '';
  if (newText && newText !== lastText) {
    lastText = newText;
    lastTextEl.textContent = newText;
    lastTextEl.classList.add('flash');
    setTimeout(() => lastTextEl.classList.remove('flash'), 900);
  }

  isPaused = data.paused;
  if (isPaused) {
    btnPause.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume`;
    btnPause.classList.add('btn-primary'); btnPause.classList.remove('btn-ghost');
  } else {
    btnPause.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`;
    btnPause.classList.remove('btn-primary'); btnPause.classList.add('btn-ghost');
  }
}

function updateHistory(items) {
  if (!items || items.length === historyCount) return;
  historyCount = items.length;
  historyCountEl.textContent = `${items.length} item${items.length !== 1 ? 's' : ''}`;

  if (items.length === 0) {
    historyList.innerHTML = `<li class="history-empty">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <p>No text recognised yet</p></li>`;
    return;
  }

  const reversed = [...items].reverse();
  historyList.innerHTML = reversed.map(item => {
    const t    = new Date(item.ts * 1000);
    const time = t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const src  = item.source || 'camera';
    const conf = item.confidence ? ` · ${(item.confidence * 100).toFixed(0)}%` : '';
    return `<li class="history-item" onclick="speakText('${escHtml(item.text)}')" title="Click to speak">
      <div style="flex:1">
        <div class="history-item-text">${escHtml(item.text)}</div>
        <div class="history-item-meta">${time}${conf}</div>
      </div>
      <span class="history-item-source ${src}">${src}</span>
    </li>`;
  }).join('');
}

// ── Quality helpers ───────────────────────────────────────────────────────────
function qualityLabel(q) {
  return { OK:'✓ Scanning', TOO_DARK:'🌑 Too dark', TOO_BRIGHT:'☀ Too bright',
    BLURRY:'〰 Blurry', MOVE_CLOSER:'🔍 Move closer', MOVE_BACK:'↔ Move back',
    NO_CAMERA:'📷 No camera', STARTING:'⏳ Starting', NO_BRAILLE:'🔍 Scanning' }[q] || q;
}
function qualityBadgeClass(q, paused) {
  if (paused)           return 'badge badge-paused';
  if (q === 'OK')       return 'badge badge-ok';
  if (q === 'STARTING') return 'badge badge-starting';
  if (q === 'NO_CAMERA') return 'badge badge-error';
  return 'badge badge-warning';
}
function qualityColour(q) {
  if (q === 'OK') return 'var(--green)';
  if (q === 'NO_CAMERA' || q === 'STARTING') return 'var(--text3)';
  return 'var(--orange)';
}

// ── TTS (browser Web Speech — works on Render; server pyttsx3 is local-only) ─
function getTtsRate() {
  const el = document.getElementById('tts-rate');
  return el ? parseInt(el.value, 10) : 175;
}

function speakInBrowser(text) {
  if (!text || !('speechSynthesis' in window)) return false;
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = Math.max(0.5, Math.min(2, getTtsRate() / 175));
    const vid = voiceSelect?.value;
    if (vid) {
      const voices = speechSynthesis.getVoices();
      const v = voices.find(x => x.voiceURI === vid || x.name === vid);
      if (v) u.voice = v;
    }
    speechSynthesis.speak(u);
    return true;
  } catch (e) {
    console.warn('Browser TTS failed', e);
    return false;
  }
}

// ── TTS controls ──────────────────────────────────────────────────────────────
async function togglePause() {
  const res  = await fetch('/api/pause', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paused: !isPaused }),
  });
  const data = await res.json();
  isPaused = data.paused;
  showToast(data.paused ? '⏸ Scanning paused' : '▶ Scanning resumed');
}

async function clearHistory() {
  if (!confirm('Clear all session history?')) return;
  await fetch('/api/clear', { method: 'POST' });
  historyCount = 0; lastText = '';
  lastTextEl.textContent = 'Waiting for Braille…';
  aiBadge.style.display = 'none';
  historyList.innerHTML = `<li class="history-empty"><p>No text recognised yet</p></li>`;
  historyCountEl.textContent = '0 items';
  setConfidence(0);
  showToast('🗑 History cleared');
}

async function speakManual() {
  const input = document.getElementById('manual-input');
  const text  = input.value.trim();
  if (!text) return;
  speakInBrowser(text);
  fetch('/api/speak', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  }).catch(() => {});
  input.value = '';
  showToast('🔊 Speaking: ' + text.slice(0, 40));
}

async function speakHistory() {
  try {
    const res = await fetch('/api/history');
    const items = await res.json();
    const full = items.map(i => i.text).filter(Boolean).join('. ');
    if (full) speakInBrowser(full);
  } catch (e) { /* ignore */ }
  fetch('/api/speak_history', { method: 'POST' }).catch(() => {});
  showToast('📖 Reading full history…');
}

async function speakLast() {
  if (!lastText) { showToast('Nothing to repeat yet'); return; }
  speakInBrowser(lastText);
  fetch('/api/speak', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: lastText }),
  }).catch(() => {});
  showToast('🔁 Repeating last text');
}

async function speakText(text) {
  speakInBrowser(text);
  fetch('/api/speak', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  }).catch(() => {});
  showToast('🔊 Speaking…');
}

function copyLast() {
  if (!lastText) { showToast('Nothing to copy yet'); return; }
  navigator.clipboard.writeText(lastText)
    .then(() => showToast('📋 Copied to clipboard'))
    .catch(() => showToast('❌ Copy failed'));
}

async function updateTTS() {
  const rate     = parseInt(document.getElementById('tts-rate').value);
  const volume   = parseInt(document.getElementById('tts-vol').value) / 100;
  const voice_id = voiceSelect.value || null;
  await fetch('/api/tts_settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rate, volume, voice_id }),
  });
}

function fillVoiceSelect(voices) {
  if (!voiceSelect || !voices.length) return;
  voiceSelect.innerHTML = '<option value="">System default</option>';
  voices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.name + (v.lang ? ` (${v.lang})` : '');
    voiceSelect.appendChild(opt);
  });
}

async function loadVoices() {
  try {
    const res  = await fetch('/api/voices');
    const data = await res.json();
    const voices = data.voices || [];
    if (voices.length) {
      fillVoiceSelect(voices);
      return;
    }
  } catch (e) { /* server voices unavailable on Render */ }

  if ('speechSynthesis' in window) {
    const loadBrowserVoices = () => {
      const bv = speechSynthesis.getVoices();
      if (!bv.length) return;
      fillVoiceSelect(bv.map(v => ({
        id: v.voiceURI,
        name: v.name,
        lang: v.lang || '',
      })));
    };
    loadBrowserVoices();
    speechSynthesis.onvoiceschanged = loadBrowserVoices;
  }
}

async function checkHealth() {
  try {
    const res  = await fetch('/api/health');
    const data = await res.json();
    if (data.groq) indGroq?.classList.add('active', 'ind-active');
    if (data.cnn)  indCnn?.classList.add('active', 'ind-active');
  } catch (e) { /* offline */ }
}

// ── Demo ──────────────────────────────────────────────────────────────────────
async function runDemo(image) {
  showToast('🧪 Running demo…');
  try {
    const res  = await fetch('/api/demo?image=' + encodeURIComponent(image));
    const data = await res.json();
    if (data.frame) {
      camImg.src = 'data:image/jpeg;base64,' + data.frame;
      camImg.style.display = 'block';
      noCamera.style.display = 'none';
    }
    if (data.text) {
      lastText = data.text;
      lastTextEl.textContent = data.text;
      lastTextEl.classList.add('flash');
      setTimeout(() => lastTextEl.classList.remove('flash'), 900);
      setConfidence(data.confidence || 0);
      showToast(`✅ Demo: ${data.text}`);
      speakInBrowser(data.text);
    } else {
      showToast('⚠ Demo: no Braille detected');
    }
    pollHistory();
  } catch (e) { showToast('❌ Demo failed'); }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

// ── Escape HTML ───────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePause(); }
  if (e.code === 'KeyR')  speakLast();
  if (e.code === 'KeyH')  speakHistory();
  if (e.code === 'KeyC')  copyLast();
  if (e.code === 'KeyU')  triggerUpload();
  if (e.code === 'KeyD')  runDemo('hello');
});

// ── Init ──────────────────────────────────────────────────────────────────────
pollFrame();
pollStatus();
pollHistory();
loadVoices();
checkHealth();
showToast('Click Allow Camera or Upload an image to start');
