"""
frame_analyzer.py
-----------------
Analyses a single camera frame for:
  1. Image quality (brightness, blur)
  2. Whether circular dot-like features consistent with braille are present
  3. Distance hint (move closer / move back)

Key fix over v1: uses circularity filter (4π·A/P²) so face texture, fabric,
and walls don't get counted as braille dots.
"""

import cv2
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────
BRIGHTNESS_LOW        = 30     # mean pixel value → too dark
BRIGHTNESS_HIGH       = 235    # mean pixel value → too bright
BLUR_THRESHOLD        = 12.0   # Laplacian variance → blurry (relaxed for paper + hand-drawn)
MIN_DOT_AREA          = 8      # px² — allow tiny embossed dots
MAX_DOT_AREA          = 60000  # px² — allow large hand-drawn dots
MIN_CIRCULARITY       = 0.30   # relaxed — hand-drawn dots aren't perfect circles
MIN_SOLIDITY          = 0.45   # relaxed
MIN_DOTS_FOR_BRAILLE  = 3      # minimum uniform circular dots (even 1 cell = 1-6 dots)
IDEAL_DOT_COUNT_LOW   = 3
IDEAL_DOT_COUNT_HIGH  = 300
SIZE_UNIFORMITY       = 0.70   # more tolerant of size variation


def analyze_frame(frame: np.ndarray) -> dict:
    """
    Analyse *frame* (BGR numpy array) and return:
        {
          "quality":          str,   # OK | TOO_DARK | TOO_BRIGHT | BLURRY
                                     #   | MOVE_CLOSER | MOVE_BACK | NO_BRAILLE
          "braille_detected": bool,
          "dot_count":        int,
          "dot_regions":      list,  # [(x,y,w,h), ...]
          "annotated":        np.ndarray,
          "gray":             np.ndarray,
        }
    """
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 1. Brightness ─────────────────────────────────────────────────────────
    mean_brightness = float(np.mean(gray))
    if mean_brightness < BRIGHTNESS_LOW:
        _draw_overlay(annotated, "TOO_DARK", (100, 100, 100))
        return _result("TOO_DARK", False, 0, [], annotated, gray)
    if mean_brightness > BRIGHTNESS_HIGH:
        _draw_overlay(annotated, "TOO_BRIGHT", (200, 200, 200))
        return _result("TOO_BRIGHT", False, 0, [], annotated, gray)

    # ── 2. Blur ───────────────────────────────────────────────────────────────
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < BLUR_THRESHOLD:
        _draw_overlay(annotated, "BLURRY", (150, 150, 150))
        return _result("BLURRY", False, 0, [], annotated, gray)

    # ── 3. Circular dot detection ─────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=4,   # softer edges for hand-drawn pen/pencil dots
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dot_regions = []
    radii = []
    h_img, w_img = frame.shape[:2]
    img_area = h_img * w_img
    # Adaptive area limits: 0.005% – 4% of image area
    adaptive_min = max(MIN_DOT_AREA, int(img_area * 0.00005))
    adaptive_max = min(MAX_DOT_AREA, int(img_area * 0.04))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (adaptive_min < area < adaptive_max):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1:
            continue
        if area / hull_area < MIN_SOLIDITY:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(1, min(w, h)) > 1.5:
            continue
        r = float(np.sqrt(area / np.pi))
        dot_regions.append((x, y, w, h))
        radii.append(r)

    # Keep only dots with similar size (real braille embossing)
    if radii:
        med_r = float(np.median(radii))
        lo, hi = med_r * (1 - SIZE_UNIFORMITY), med_r * (1 + SIZE_UNIFORMITY)
        filtered = []
        for (x, y, w, h), r in zip(dot_regions, radii):
            if lo <= r <= hi:
                filtered.append((x, y, w, h))
                # Colour by size relative to median: green=good, yellow=borderline
                colour = (255, 255, 255)
                cv2.circle(annotated, (x + w // 2, y + h // 2), max(4, w // 2), colour, 2)
        dot_regions = filtered

    dot_count = len(dot_regions)
    braille_detected = dot_count >= MIN_DOTS_FOR_BRAILLE

    # ── 4. Quality / distance hint ────────────────────────────────────────────
    if not braille_detected:
        quality = "NO_BRAILLE"
        colour  = (120, 120, 120)
    elif dot_count < IDEAL_DOT_COUNT_LOW:
        quality = "MOVE_CLOSER"
        colour  = (200, 200, 200)
    elif dot_count > IDEAL_DOT_COUNT_HIGH:
        quality = "MOVE_BACK"
        colour  = (200, 200, 200)
    else:
        quality = "OK"
        colour  = (255, 255, 255)

    _draw_overlay(annotated, quality, colour,
                  dot_count=dot_count, brightness=mean_brightness, blur=lap_var)
    return _result(quality, braille_detected, dot_count, dot_regions, annotated, gray)


def _result(quality, detected, count, regions, annotated, gray):
    return {
        "quality":          quality,
        "braille_detected": detected,
        "dot_count":        count,
        "dot_regions":      regions,
        "annotated":        annotated,
        "gray":             gray,
    }


def _draw_overlay(frame, quality, colour, dot_count=0, brightness=0.0, blur=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    labels = {
        "OK":          "✓ Braille detected — scanning",
        "TOO_DARK":    "⚠ Too dark — improve lighting",
        "TOO_BRIGHT":  "⚠ Too bright — reduce glare",
        "BLURRY":      "⚠ Blurry — hold camera steady",
        "MOVE_CLOSER": "→ Move closer to the page",
        "MOVE_BACK":   "← Move back a little",
        "NO_BRAILLE":  "  Scanning — no braille detected",
    }
    label = labels.get(quality, quality)
    cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    if dot_count or brightness:
        info = f"dots:{dot_count}  brightness:{brightness:.0f}  sharpness:{blur:.0f}"
        cv2.putText(frame, info, (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1)
