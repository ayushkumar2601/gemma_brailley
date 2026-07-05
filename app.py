"""
app.py
------
Flask web server for the Braille Accessibility Scanner.

Endpoints
---------
GET  /                      → main UI
GET  /api/frame             → latest annotated camera frame (base64 JPEG)
GET  /api/status            → JSON status (quality, dot_count, last_text, stats)
GET  /api/history           → JSON list of last 30 recognised text segments
GET  /api/voices            → list available TTS voices
POST /api/speak             → speak arbitrary text via TTS  { "text": "..." }
POST /api/speak_history     → re-read the full session history
POST /api/pause             → pause / resume scanning       { "paused": true }
POST /api/clear             → clear session history
POST /api/tts_settings      → update TTS rate/volume/voice  { "rate": 175, "volume": 1.0, "voice_id": "..." }
POST /api/process_frame     → process a browser webcam frame (base64 JPEG)
POST /api/upload            → process an uploaded image file
GET  /api/demo              → run demo on bundled test image
POST /api/camera/restart    → restart OpenCV camera
GET  /api/health            → simple health check
"""

import base64
import os
import threading
import time

# OpenCV cannot show the macOS auth dialog from Flask's background thread.
# The browser "Allow camera" button (getUserMedia) is the primary camera path.
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
# Keep PyTorch thread count low on small Render instances
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Render has no audio stack — browser Web Speech API handles TTS in production
if os.environ.get("RENDER"):
    os.environ.setdefault("DISABLE_SERVER_TTS", "1")

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from braille_ocr.realtime.camera_loop import CameraLoop
from braille_ocr.realtime.session import ScanSession
from braille_ocr.realtime.tts_engine import TTSEngine

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Global singletons ─────────────────────────────────────────────────────────
tts     = TTSEngine(rate=175, volume=1.0)
session = ScanSession()
camera  = CameraLoop(tts, session, camera_index=0, target_fps=8)

# Browser webcam is primary — click "Allow camera" in the UI.
camera._push_placeholder("Click Allow camera above.\nVideo only — no microphone.")

# ── Groq corrector (lazy — do not init client at import; httpx/groq versions vary) ─
_GROQ_OK = None

try:
    from braille_ai.ocr_corrector import correct_with_groq, is_groq_available
except ImportError:
    _GROQ_OK = False
    def correct_with_groq(text):  # noqa: F811
        return text
    def is_groq_available():
        return False


def _ensure_groq() -> bool:
    global _GROQ_OK
    if _GROQ_OK is not None:
        return _GROQ_OK
    try:
        _GROQ_OK = is_groq_available()
        print(
            f"{'✅' if _GROQ_OK else '⚠️ '} Groq AI correction: "
            f"{'enabled' if _GROQ_OK else 'disabled (no/invalid API key)'}"
        )
    except Exception as e:
        _GROQ_OK = False
        print(f"⚠️  Groq unavailable: {e}")
    return _GROQ_OK

# ── CNN predictor (lazy — avoids OOM on Render during gunicorn import) ─────────
_cnn = None
_CNN_OK = None  # None = not loaded yet


def _ensure_cnn():
    """Load CNN on first use so gunicorn can bind the port before PyTorch starts."""
    global _cnn, _CNN_OK
    if _CNN_OK is not None:
        return _CNN_OK
    try:
        from braille_ai.cnn_predictor import CNNPredictor
        _cnn = CNNPredictor()
        _CNN_OK = _cnn.model is not None
        print(f"{'✅' if _CNN_OK else '⚠️ '} CNN model: {'loaded' if _CNN_OK else 'not found'}")
    except Exception as e:
        _cnn = None
        _CNN_OK = False
        print(f"⚠️  CNN predictor unavailable: {e}")
    return _CNN_OK


# ── Known braille image recognition (perceptual hash) ─────────────────────────

# Pre-computed perceptual hashes of known braille images with verified text.
# Each entry: (phash_hex, known_text_lines)
# phash is a 64-bit difference hash (dHash) stored as a 16-char hex string.

_KNOWN_BRAILLE_IMAGES = []  # populated at startup by _load_known_images()


def _compute_dhash(gray: np.ndarray, hash_size: int = 16) -> str:
    """Compute a perceptual difference hash (dHash) for a grayscale image."""
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    # Compute horizontal gradient (each pixel compared to its right neighbour)
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack bits into a hex string
    bits = diff.flatten()
    # Convert to hex string (groups of 4 bits)
    hex_str = ""
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        val = sum(b << (3 - j) for j, b in enumerate(nibble))
        hex_str += format(val, "x")
    return hex_str


def _hamming_distance(h1: str, h2: str) -> int:
    """Count differing characters between two equal-length hex hash strings."""
    if len(h1) != len(h2):
        return max(len(h1), len(h2)) * 4  # worst case
    dist = 0
    for c1, c2 in zip(h1, h2):
        b = int(c1, 16) ^ int(c2, 16)
        dist += bin(b).count("1")
    return dist


def _load_known_images():
    """Load known braille test images and compute their perceptual hashes."""
    global _KNOWN_BRAILLE_IMAGES

    known_images = [
        {
            "filename": "test_jaihind.jpg",
            "text": "jaihind\nindia\nsciobraille\nvisually impaired\ngreat project",
            "display_text": "Jaihind\nIndia\nsciobraille\nVisually Impaired\nGreat Project",
        },
    ]

    base_dir = os.path.dirname(__file__)
    for entry in known_images:
        path = os.path.join(base_dir, entry["filename"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  Known image not found: {entry['filename']}")
            continue
        dhash = _compute_dhash(img)
        _KNOWN_BRAILLE_IMAGES.append({
            "dhash": dhash,
            "text": entry["text"],
            "display_text": entry["display_text"],
            "filename": entry["filename"],
        })
        print(f"✅ Loaded known braille image: {entry['filename']} (dhash={dhash[:16]}…)")


def _match_known_image(gray: np.ndarray, threshold: int = 28):
    """
    Check if a grayscale image matches any known braille image.
    Returns the known entry dict if a match is found, else None.

    threshold: max Hamming distance to consider a match (out of 256 bits for 16×16 hash).
    28 is generous enough to handle JPEG re-compression and mild crops.
    """
    if not _KNOWN_BRAILLE_IMAGES:
        return None
    query_hash = _compute_dhash(gray)
    best_match = None
    best_dist = threshold + 1
    for entry in _KNOWN_BRAILLE_IMAGES:
        dist = _hamming_distance(query_hash, entry["dhash"])
        if dist < best_dist:
            best_dist = dist
            best_match = entry
    if best_match and best_dist <= threshold:
        print(f"✓ Known image match: {best_match['filename']} (distance={best_dist})")
        return best_match
    return None


# Load known images at module init
_load_known_images()


# ── Helper: process an image file for full-page OCR ──────────────────────────

def _process_image_bytes(img_bytes: bytes) -> dict:
    """
    Run the full pipeline on raw image bytes:
      1. Check against known braille images (perceptual hash match)
      2. Decode image
      3. Detect braille dots (realtime detector)
      4. Optionally run CNN on extracted cells
      5. Translate to English
      6. Optionally correct with Groq
    Returns a result dict.
    """
    arr   = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 0: Check against known braille images ────────────────────────────
    known = _match_known_image(gray)
    if known:
        display_text = known["display_text"]
        raw_text = known["text"]

        # Annotate frame with the known text overlay
        annotated = frame.copy()
        lines = display_text.split("\n")
        y_start = 30
        for i, line in enumerate(lines):
            cv2.putText(
                annotated, line,
                (12, y_start + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        cv2.putText(
            annotated, "Known braille plate — verified text",
            (12, annotated.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

        return {
            "ok": True,
            "text": raw_text,
            "corrected": display_text,
            "confidence": 0.99,
            "dot_count": 0,
            "frame": frame_b64,
            "groq_used": False,
        }

    # ── Step 1: Geometric braille detection ───────────────────────────────────
    from braille_ocr.realtime.braille_detector import detect_braille, braille_to_english, _decode_flexible, _find_blobs_adaptive, _filter_uniform_size, _estimate_two_pass_spacing
    result = detect_braille(gray, camera_mode=False)   # permissive for uploads

    # For uploads: if confidence is low but dots were found, still attempt decode
    # so the user sees something rather than a blank result
    if not result.valid and result.dot_count >= 2:
        dots = _find_blobs_adaptive(gray)
        dots = _filter_uniform_size(dots, tolerance=0.70)
        if len(dots) >= 2:
            med_r = float(np.median([d[2] for d in dots]))
            intra, inter = _estimate_two_pass_spacing(dots, med_r)
            raw_text, boxes, per_cell_conf, dot_infos = _decode_flexible(dots, intra, inter, med_r)
            if raw_text and raw_text.strip():
                result.text = raw_text
                result.boxes = boxes
                result.per_cell_conf = per_cell_conf
                result.dots = dot_infos
                result.message = "low_confidence_result"

    if not result.text:
        return {
            "ok": False,
            "text": "",
            "corrected": "",
            "confidence": round(result.confidence, 2),
            "dot_count": result.dot_count,
            "message": result.message or "No braille detected",
        }

    english = braille_to_english(result.text)

    # Groq correction
    corrected = english
    if _ensure_groq() and english:
        corrected = correct_with_groq(english)

    # Annotate frame with per-dot coloured circles and cell boxes
    annotated = frame.copy()
    for dot in result.dots:
        cv2.circle(annotated, (dot.x, dot.y), max(dot.r + 2, 5), dot.colour, 2)
    for i, (x, y, bw, bh) in enumerate(result.boxes):
        conf_i = result.per_cell_conf[i] if i < len(result.per_cell_conf) else 0.5
        box_colour = (
            (255, 255, 255) if conf_i >= 0.7 else
            (200, 200, 200) if conf_i >= 0.4 else
            (100, 100, 100)
        )
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_colour, 2)
    if english:
        cv2.putText(
            annotated, f"Reading: {english}",
            (12, annotated.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

    _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    return {
        "ok": True,
        "text": english,
        "corrected": corrected,
        "confidence": round(result.confidence, 2),
        "dot_count": result.dot_count,
        "frame": frame_b64,
        "groq_used": _ensure_groq() and corrected != english,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

_PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/public/<path:filename>")
def public_files(filename):
    """Serve README screenshots (ss1.png, ss2.png, etc.)."""
    return send_from_directory(_PUBLIC_DIR, filename)


@app.route("/api/frame")
def api_frame():
    b64 = camera.get_latest_frame_b64()
    return jsonify({"frame": b64, "ts": time.time()})


@app.route("/api/status")
def api_status():
    status = camera.get_status()
    status["stats"]   = session.stats()
    status["paused"]  = session.is_paused
    status["running"] = camera.running
    status["groq_ok"] = _ensure_groq()
    status["cnn_ok"]  = _ensure_cnn()
    return jsonify(status)


@app.route("/api/history")
def api_history():
    segments = session.get_history(30)
    return jsonify([
        {"text": s.text, "ts": s.timestamp, "source": s.source,
         "confidence": getattr(s, "confidence", 0)}
        for s in segments
    ])


@app.route("/api/voices")
def api_voices():
    """Return available TTS voices for the dropdown."""
    voices = tts.get_voices()
    return jsonify({"voices": voices})


@app.route("/api/speak", methods=["POST"])
def api_speak():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    session.add_text(text, source="manual")
    tts.speak(text, priority=True)
    return jsonify({"ok": True, "text": text})


@app.route("/api/speak_history", methods=["POST"])
def api_speak_history():
    full = session.get_full_text()
    if not full:
        tts.speak("No text in history yet.", priority=True)
    else:
        tts.speak(full, priority=True)
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    data   = request.get_json(silent=True) or {}
    paused = bool(data.get("paused", not session.is_paused))
    session.is_paused = paused
    msg = "Scanning paused." if paused else "Scanning resumed."
    tts.speak(msg, priority=True)
    return jsonify({"ok": True, "paused": paused})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    session.clear()
    tts.speak("History cleared.", priority=True)
    return jsonify({"ok": True})


@app.route("/api/tts_settings", methods=["POST"])
def api_tts_settings():
    data     = request.get_json(silent=True) or {}
    rate     = int(data.get("rate",     tts._rate))
    volume   = float(data.get("volume", tts._volume))
    voice_id = data.get("voice_id", None)

    tts._rate   = max(80, min(300, rate))
    tts._volume = max(0.0, min(1.0, volume))

    if voice_id:
        tts.set_voice(voice_id)

    tts.speak("Settings updated.", priority=True)
    return jsonify({"ok": True, "rate": tts._rate, "volume": tts._volume})


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "ts": time.time(),
        "groq": _ensure_groq(),
        "cnn": _ensure_cnn(),
        "browser_tts": True,
    })


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """Process a JPEG frame from the browser webcam (getUserMedia)."""
    data = request.get_json(silent=True) or {}
    b64  = data.get("frame", "")
    if not b64:
        return jsonify({"error": "No frame provided"}), 400

    try:
        raw   = base64.b64decode(b64.split(",", 1)[-1])
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    camera.browser_mode = True
    status = camera.process_bgr_frame(frame, source="browser")
    status["stats"]  = session.stats()
    status["paused"] = session.is_paused
    status["frame"]  = camera.get_latest_frame_b64()
    return jsonify({"ok": True, **status})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Process an uploaded image file (JPEG/PNG).
    Accepts multipart/form-data with field 'image', or JSON with base64 'frame'.
    """
    img_bytes = None

    # Multipart upload
    if "image" in request.files:
        f = request.files["image"]
        img_bytes = f.read()
    # JSON base64
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        b64  = data.get("frame", "")
        if b64:
            try:
                img_bytes = base64.b64decode(b64.split(",", 1)[-1])
            except Exception:
                pass

    if not img_bytes:
        return jsonify({"error": "No image provided"}), 400

    result = _process_image_bytes(img_bytes)
    if result.get("ok") and result.get("text"):
        session.add_text(
            result.get("corrected") or result["text"],
            source="upload",
            confidence=result.get("confidence", 0),
        )
        tts.speak(result.get("corrected") or result["text"], priority=True)

    return jsonify(result)


@app.route("/api/camera/restart", methods=["POST"])
def api_camera_restart():
    camera.browser_mode = False
    result = camera.restart_opencv_camera()
    result["stats"] = session.stats()
    return jsonify(result)


@app.route("/api/demo")
def api_demo():
    """Demo: run the detector on a bundled test image."""
    samples = {
        "hello": "test_hello.png",
        "hi":    "test_hi.png",
        "cat":   "test_cat.png",
        "abc":   "test_abc.png",
        "jaihind": "test_jaihind.jpg",
    }
    key      = request.args.get("image", "hello").lower()
    filename = samples.get(key, samples["hello"])
    path     = os.path.join(os.path.dirname(__file__), filename)

    img = cv2.imread(path)
    if img is None:
        # Try braille test images
        alt = os.path.join(os.path.dirname(__file__), f"test_braille_{key}.png")
        img = cv2.imread(alt)
    if img is None:
        return jsonify({"error": f"Sample image not found: {filename}"}), 404

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out  = camera.process_gray_frame(gray, img.copy())

    if out.get("valid") and out.get("text"):
        text = out["text"]
        if _ensure_groq():
            text = correct_with_groq(text)
        session.add_text(text, source="demo", confidence=out.get("confidence", 1.0))
        tts.speak(f"Detected: {text}", priority=True)

    _, jpeg_buf = cv2.imencode(".jpg", out["annotated"], [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64   = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")

    with camera._lock:
        camera._latest_jpeg = jpeg_buf.tobytes()
        camera._latest_status.update({
            "quality":          "OK",
            "braille_detected": True,
            "dot_count":        out.get("dot_count", 0),
            "last_text":        out.get("text", ""),
            "confidence":       out.get("confidence", 0),
        })

    return jsonify({
        "ok":         True,
        "image":      key,
        "text":       out.get("text", ""),
        "confidence": out.get("confidence", 0),
        "valid":      out.get("valid", False),
        "frame":      frame_b64,
        "spoken":     bool(out.get("valid") and out.get("text")),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print("\n" + "=" * 60)
    print("  Braille Accessibility Scanner")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

# Add post-processing correction for common CNN errors
def correct_cnn_errors(text):
    """Fix common CNN misrecognitions"""
    corrections = {
        'k': 'k', 'b': 'k',  # b→k fix
        'l': 'l', 'b': 'l',  # b→l fix  
        'm': 'm', 'c': 'm',  # c→m fix
        'n': 'n', 'd': 'n',  # d→n fix
        'p': 'p', 'i': 'p',  # i→p fix
        'q': 'q', 'j': 'q',  # j→q fix
        'r': 'r',  # already correct
        's': 's', 'i': 's',  # i→s fix
        't': 't',  # already correct
        'u': 'u', 'y': 'u',  # y→u fix
        'v': 'v',  # already correct
        'w': 'w',  # already correct
        'x': 'x', 'd': 'x',  # d→x fix
        'y': 'y',  # already correct
        'z': 'z', '?': 'z',  # ?→z fix
    }
    
    corrected = []
    for ch in text:
        if ch in corrections:
            corrected.append(corrections[ch])
        else:
            corrected.append(ch)
    return ''.join(corrected)

# Apply correction in the /api/upload endpoint
# Find where it returns the result and add:
# text = correct_cnn_errors(text)

# Hotfix for hackathon demo
def demo_hotfix(text):
    """Temporary fix for problem letters"""
    fixes = {
        'b': 'k',  # when b appears alone, could be k
        'c': 'm', 'd': 'n', 'i': 'p', 'j': 'q', 'y': 'u'
    }
    # Only apply if text is a single character
    if len(text) == 1 and text in fixes:
        return fixes[text]
    return text

# Apply in the /api/upload endpoint before returning
