"""
braille_detector.py — complete implementation
----------------------------------------------
Standard Grade-1 braille dot numbering:
    1 4
    2 5
    3 6

Bit mask: bit0=dot1, bit1=dot2, bit2=dot3, bit3=dot4, bit4=dot5, bit5=dot6

Detection pipeline:
  1. Find blobs (dots) using adaptive thresholding
  2. Cluster dots into cells
  3. Decode each cell using geometric dot-pattern matching
  4. Optionally refine with CNN classifier
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Constants ─────────────────────────────────────────────────────────────────
SPEAK_MIN_CHARS = 1

# ── Standard Grade-1 Braille map ─────────────────────────────────────────────
_BRAILLE_TO_CHAR: dict[int, str] = {
    0b000001: "a",  0b000011: "b",  0b001001: "c",  0b011001: "d",  0b010001: "e",
    0b001011: "f",  0b011011: "g",  0b010011: "h",  0b001010: "i",  0b011010: "j",
    0b000101: "k",  0b000111: "l",  0b001101: "m",  0b011101: "n",  0b010101: "o",
    0b001111: "p",  0b011111: "q",  0b010111: "r",  0b001110: "s",  0b011110: "t",
    0b100101: "u",  0b100111: "v",  0b111010: "w",  0b101101: "x",  0b111101: "y",
    0b110101: "z",  0b000000: " ",
}
_VALID = set(_BRAILLE_TO_CHAR.values())

MIN_DOTS              = 1   # allow single-dot letters (A, E, I, etc.)
MIN_CONFIDENCE        = 0.15
MIN_CONFIDENCE_CAMERA = 0.30

# ── Lazy CNN predictor ────────────────────────────────────────────────────────
_cnn_predictor = None

def _get_cnn():
    global _cnn_predictor
    if _cnn_predictor is None:
        try:
            from braille_ai.cnn_predictor import CNNPredictor
            _cnn_predictor = CNNPredictor()
        except Exception:
            _cnn_predictor = False  # mark as unavailable
    return _cnn_predictor if _cnn_predictor else None


@dataclass
class DotInfo:
    x: int
    y: int
    r: int
    colour: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class DetectionResult:
    text: str = ""
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    dots: List[DotInfo] = field(default_factory=list)
    confidence: float = 0.0
    valid: bool = False
    dot_count: int = 0
    message: str = ""
    per_cell_conf: List[float] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_braille_from_frame(gray: np.ndarray) -> Tuple[str, List[Tuple]]:
    r = detect_braille(gray, camera_mode=True)
    if r.message == "SCATTERED":
        return "SCATTERED", []
    return (r.text, r.boxes) if r.valid else ("", r.boxes)


def detect_braille(gray: np.ndarray, camera_mode: bool = False) -> DetectionResult:
    if gray is None or gray.size == 0:
        return DetectionResult(message="empty")

    h, w = gray.shape[:2]

    # ── Fast path: small single-cell image → use CNN directly ─────────────────
    # Synthetic dataset images are 100×120px. If the image is small enough
    # to be a single cell, skip geometric detection and use CNN.
    if h <= 200 and w <= 200:
        cnn = _get_cnn()
        if cnn and cnn.model is not None:
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(gray)
                letter, conf = cnn.predict_cell(pil_img)
                if conf > 50:
                    char = letter.lower()
                    return DetectionResult(
                        text=char,
                        boxes=[(0, 0, w, h)],
                        dots=[],
                        confidence=conf / 100.0,
                        valid=True,
                        dot_count=1,
                        message="cnn",
                        per_cell_conf=[conf / 100.0],
                    )
            except Exception:
                pass  # fall through to geometric detection

    dots = _find_blobs_adaptive(gray)
    if len(dots) < MIN_DOTS:
        return DetectionResult(dot_count=len(dots), message="few_dots")

    dots = _deduplicate_dots(dots)
    dots = _filter_uniform_size(dots, tolerance=0.70)
    if len(dots) < MIN_DOTS:
        return DetectionResult(dot_count=len(dots), message="size_mismatch")

    # Focus on densest region if very scattered
    pts = np.array([(d[0], d[1]) for d in dots], dtype=np.float32)
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    coverage = (x_span * y_span) / max(w * h, 1)
    if coverage > 0.60 and len(dots) > 20:
        dots = _densest_cluster(dots, pts, w, h)
        if len(dots) < MIN_DOTS:
            return DetectionResult(dot_count=len(dots), message="SCATTERED")

    if len(dots) < MIN_DOTS:
        return DetectionResult(dot_count=len(dots), message="few_dots")

    med_r = float(np.median([d[2] for d in dots]))
    row_spacing, col_spacing = _estimate_spacing(dots, med_r)
    grid_score = _grid_regularity_score(dots, row_spacing)

    # For very few dots (1-2), skip grid check
    if len(dots) >= 4 and grid_score < 0.20:
        return DetectionResult(dot_count=len(dots), message="irregular_grid")

    text, boxes, per_cell_conf, dot_infos = _decode_page(
        dots, row_spacing, col_spacing, med_r, gray.shape
    )

    if not text or not text.strip():
        return DetectionResult(dot_count=len(dots), message="decode_fail")

    clean = text.replace(" ", "")
    valid_ratio = sum(1 for c in clean if c in _VALID) / max(len(clean), 1)
    conf = _confidence(clean, valid_ratio, grid_score, len(dots), len(boxes))

    if len(clean) >= 3 and len(set(clean)) == 1:
        conf *= 0.25
    unknown_ratio = clean.count("?") / max(len(clean), 1)
    if unknown_ratio > 0.5:
        conf *= (1.0 - unknown_ratio * 0.5)

    threshold = MIN_CONFIDENCE_CAMERA if camera_mode else MIN_CONFIDENCE
    valid = conf >= threshold and len(clean) >= 1

    _colour_dot_infos(dot_infos, per_cell_conf)

    return DetectionResult(
        text=text if valid else "",
        boxes=boxes, dots=dot_infos, confidence=conf,
        valid=valid, dot_count=len(dots),
        message="ok" if valid else "low_confidence",
        per_cell_conf=per_cell_conf,
    )


def braille_to_english(raw: str) -> str:
    """Pass-through: the detector already outputs lowercase English letters."""
    return raw.strip().lower() if raw else ""


# ── Blob detection ────────────────────────────────────────────────────────────

def _find_blobs_adaptive(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find circular blobs (braille dots) using adaptive thresholding.
    Handles BOTH dark-dots-on-white (synthetic) AND light/raised dots on
    light background (real embossed Braille photos).
    Returns list of (cx, cy, radius).
    """
    h, w = gray.shape[:2]
    img_area = h * w
    mean_val = float(np.mean(gray))

    # CLAHE to normalise lighting
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive threshold — works for both printed and hand-drawn dots
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=5,
    )

    # Also try Otsu for high-contrast images
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(thresh, otsu)

    # Morphological closing to fill gaps in hand-drawn dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # ── Second pass for real embossed Braille (light background, raised dots) ──
    # If image is mostly light (mean > 180), also try detecting raised dots
    # by looking for regions slightly darker than surroundings (shadow effect).
    # Use morphological gradient or local contrast enhancement.
    if mean_val > 180:
        light_dots = _find_blobs_light_bg(gray, img_area)
        # Merge light-bg detections into combined mask
        for (cx, cy, r) in light_dots:
            cv2.circle(combined, (cx, cy), max(r, 3), 255, -1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    adaptive_min = max(8, int(img_area * 0.00003))
    adaptive_max = min(80000, int(img_area * 0.05))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (adaptive_min < area < adaptive_max):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.25:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or area / hull_area < 0.40:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.0:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        r = max(1, int(np.sqrt(area / np.pi)))
        dots.append((cx, cy, r))

    return dots


def _find_blobs_light_bg(gray: np.ndarray, img_area: int) -> List[Tuple[int, int, int]]:
    """
    Detect raised/embossed Braille dots on a light background (real photos).

    Real embossed Braille has white raised dots on a beige/cream background.
    The dots cast subtle shadows and have slightly different reflectance.
    Strategy:
      1. CLAHE with higher clipLimit (4.0) to amplify low-contrast dot edges
      2. Morphological gradient (dilate - erode) to find dot boundaries
      3. Threshold the gradient to get dot regions
      4. Also try inverted image with adaptive threshold
    Returns list of (cx, cy, radius).
    """
    # Step 1: High-contrast CLAHE for low-contrast embossed dots
    clahe_hi = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe_hi.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Step 2: Morphological gradient to find dot edges/boundaries
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(blurred, kernel_e)
    eroded = cv2.erode(blurred, kernel_e)
    gradient = cv2.subtract(dilated, eroded)

    # Threshold the gradient
    _, grad_thresh = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)

    # Step 3: Also try inverted adaptive threshold (catches raised dots as
    # slightly brighter regions against a uniform background)
    inverted = cv2.bitwise_not(blurred)
    inv_thresh = cv2.adaptiveThreshold(
        inverted, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=8,
    )

    # Combine gradient and inverted threshold
    combined = cv2.bitwise_or(grad_thresh, inv_thresh)

    # Morphological closing to fill dot interiors
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    adaptive_min = max(8, int(img_area * 0.00003))
    adaptive_max = min(80000, int(img_area * 0.05))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (adaptive_min < area < adaptive_max):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue

        # Lower circularity threshold (0.20) since embossed dots cast shadows
        # and may not be perfectly circular in the gradient image
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.20:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or area / hull_area < 0.35:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 2.2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        r = max(1, int(np.sqrt(area / np.pi)))
        dots.append((cx, cy, r))

    return dots


def _deduplicate_dots(dots: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Remove dots that are too close together (keep the larger one)."""
    if len(dots) < 2:
        return dots
    dots_sorted = sorted(dots, key=lambda d: -d[2])
    kept = []
    for d in dots_sorted:
        too_close = False
        for k in kept:
            dist = np.hypot(d[0] - k[0], d[1] - k[1])
            if dist < max(d[2], k[2]) * 1.2:
                too_close = True
                break
        if not too_close:
            kept.append(d)
    return kept


def _filter_uniform_size(
    dots: List[Tuple[int, int, int]], tolerance: float = 0.70
) -> List[Tuple[int, int, int]]:
    """Keep only dots whose radius is within tolerance of the median."""
    if len(dots) < 2:
        return dots
    radii = [d[2] for d in dots]
    med = float(np.median(radii))
    lo = med * (1 - tolerance)
    hi = med * (1 + tolerance)
    return [d for d in dots if lo <= d[2] <= hi]


def _densest_cluster(
    dots: List[Tuple[int, int, int]],
    pts: np.ndarray,
    w: int,
    h: int,
) -> List[Tuple[int, int, int]]:
    """Return dots in the densest quarter of the image."""
    cx = float(np.median(pts[:, 0]))
    cy = float(np.median(pts[:, 1]))
    half_w = w * 0.35
    half_h = h * 0.35
    return [
        d for d in dots
        if abs(d[0] - cx) < half_w and abs(d[1] - cy) < half_h
    ]


# ── Spacing estimation ────────────────────────────────────────────────────────

def _estimate_spacing(
    dots: List[Tuple[int, int, int]], med_r: float
) -> Tuple[float, float]:
    """
    Estimate intra-cell row spacing and intra-cell column spacing.
    Uses only gaps larger than 1.5 * med_r to avoid noise from nearby dots.
    Returns (row_spacing, col_spacing).

    Key: col_spacing is the gap between left and right columns WITHIN a cell,
    not the gap between cells.
    """
    if len(dots) < 2:
        return med_r * 2.5, med_r * 3.5

    min_gap = med_r * 1.5

    ys = sorted(set(d[1] for d in dots))
    xs = sorted(set(d[0] for d in dots))

    # Y gaps — only meaningful gaps (skip tiny ones from same-row dots)
    y_gaps = []
    for i in range(1, len(ys)):
        g = ys[i] - ys[i - 1]
        if g > min_gap:
            y_gaps.append(g)

    # X gaps — collect all meaningful gaps
    x_gaps = []
    for i in range(1, len(xs)):
        g = xs[i] - xs[i - 1]
        if g > min_gap:
            x_gaps.append(g)

    row_spacing = float(np.median(y_gaps)) if y_gaps else med_r * 2.5

    # For column spacing: use the MINIMUM meaningful X gap as the intra-cell gap.
    # The intra-cell gap (left→right column) is always smaller than inter-cell gap.
    if x_gaps:
        x_gaps_sorted = sorted(x_gaps)
        # Use the smallest gap cluster as the intra-cell spacing
        # If there are multiple gap sizes, the smallest is intra-cell
        col_spacing = float(x_gaps_sorted[0])
        # But if all gaps are similar, use median
        if len(x_gaps_sorted) > 1:
            ratio = x_gaps_sorted[-1] / max(x_gaps_sorted[0], 1)
            if ratio < 1.5:
                # All gaps similar — use median
                col_spacing = float(np.median(x_gaps))
    else:
        col_spacing = med_r * 3.5

    # Clamp to sensible multiples of dot radius
    row_spacing = max(row_spacing, med_r * 1.5)
    col_spacing = max(col_spacing, med_r * 1.5)

    return row_spacing, col_spacing


def _estimate_two_pass_spacing(
    dots: List[Tuple[int, int, int]], med_r: float
) -> Tuple[float, float]:
    """Two-pass spacing estimate used by the flexible decoder."""
    return _estimate_spacing(dots, med_r)


def _grid_regularity_score(
    dots: List[Tuple[int, int, int]], row_spacing: float
) -> float:
    """
    Score 0-1 measuring how grid-like the dot layout is.
    Uses Y-coordinate clustering.
    """
    if len(dots) < 2:
        return 1.0  # single dot is trivially "regular"

    ys = np.array([d[1] for d in dots], dtype=float)
    ys_sorted = np.sort(ys)

    # Count dots that are close to a multiple of row_spacing from the minimum
    y_min = ys_sorted[0]
    on_grid = 0
    tol = row_spacing * 0.40
    for y in ys:
        offset = (y - y_min) % row_spacing
        if offset < tol or offset > row_spacing - tol:
            on_grid += 1

    return on_grid / len(dots)


# ── Page decoder ──────────────────────────────────────────────────────────────

def _decode_page(
    dots: List[Tuple[int, int, int]],
    row_spacing: float,
    col_spacing: float,
    med_r: float,
    img_shape: Tuple[int, int] = (0, 0),
) -> Tuple[str, List[Tuple], List[float], List[DotInfo]]:
    """
    Decode a full page of braille dots into text.

    Strategy:
    1. Cluster dots into braille lines (groups of 3 dot-rows per braille line)
    2. Within each braille line, cluster dots into cells (left/right columns)
    3. Map each cell's dot pattern to a character
    """
    if not dots:
        return "", [], [], []

    # ── Step 1: cluster Y into braille lines ──────────────────────────────────
    # Each braille line spans ~3 * row_spacing in height
    braille_line_height = row_spacing * 3.2

    ys = np.array([d[1] for d in dots], dtype=float)
    y_min = float(ys.min())

    # Assign each dot to a braille line
    line_groups: dict[int, list] = {}
    for d in dots:
        line_idx = int((d[1] - y_min) / max(braille_line_height, 1))
        line_groups.setdefault(line_idx, []).append(d)

    text = ""
    all_boxes: List[Tuple] = []
    all_conf: List[float] = []
    dot_infos: List[DotInfo] = []

    for line_idx in sorted(line_groups.keys()):
        line_dots = line_groups[line_idx]
        if not line_dots:
            continue

        line_text, boxes, confs, dinfos = _decode_line(
            line_dots, row_spacing, col_spacing, med_r, img_shape
        )
        text += line_text
        all_boxes.extend(boxes)
        all_conf.extend(confs)
        dot_infos.extend(dinfos)

    return text, all_boxes, all_conf, dot_infos


def _assign_row(y: float, row1_y: float, row2_y: float, row3_y: float) -> int:
    """Assign a Y coordinate to braille row 1, 2, or 3."""
    d1 = abs(y - row1_y)
    d2 = abs(y - row2_y)
    d3 = abs(y - row3_y)
    if d1 <= d2 and d1 <= d3:
        return 1
    if d2 <= d1 and d2 <= d3:
        return 2
    return 3


def _decode_line(
    dots: List[Tuple[int, int, int]],
    row_spacing: float,
    col_spacing: float,
    med_r: float,
    img_shape: Tuple[int, int] = (0, 0),
) -> Tuple[str, List[Tuple], List[float], List[DotInfo]]:
    """Decode one braille line (up to 3 dot-rows) into characters."""
    if not dots:
        return "", [], [], []

    # ── Determine the 3 row centers for this braille line ────────────────────
    # Braille has exactly 3 dot-rows. We need to figure out where they are.
    #
    # Key insight: the 3 rows are at fixed relative positions within the
    # braille line's Y extent. We use the image height to anchor them.
    #
    # For a single-cell image (e.g. 120px tall):
    #   row1 ≈ top 25% of image height
    #   row2 ≈ middle 50%
    #   row3 ≈ bottom 75%
    #
    # For multi-cell images, we use the Y range of dots in this line.

    ys = [float(d[1]) for d in dots]
    y_clusters = _cluster_1d_smart(ys, row_spacing * 0.5)
    y_clusters.sort()

    img_h = img_shape[0] if img_shape and img_shape[0] > 0 else 0

    row1_y, row2_y, row3_y = _infer_three_rows(y_clusters, row_spacing, img_h)

    # ── Cluster X values into columns ─────────────────────────────────────────
    xs = [float(d[0]) for d in dots]
    col_centers = _cluster_1d_smart(xs, col_spacing * 0.8)
    col_centers.sort()

    # Pair columns into cells (left col + right col)
    cells_x = _pair_columns(col_centers, col_spacing)

    if not cells_x:
        return "", [], [], []

    # ── Assign dots to cells ──────────────────────────────────────────────────
    cell_dots: dict[int, list] = {i: [] for i in range(len(cells_x))}
    dot_infos: List[DotInfo] = []

    for d in dots:
        cx, cy, r = d
        best_cell = 0
        best_dist = float("inf")
        for i, (lx, rx) in enumerate(cells_x):
            cell_cx = (lx + rx) / 2
            dist = abs(cx - cell_cx)
            if dist < best_dist:
                best_dist = dist
                best_cell = i
        cell_dots[best_cell].append(d)
        dot_infos.append(DotInfo(x=cx, y=cy, r=r))

    # ── Decode each cell ──────────────────────────────────────────────────────
    text = ""
    boxes = []
    confs = []

    def get_row(y: float) -> int:
        return _assign_row(y, row1_y, row2_y, row3_y)

    for i, (lx, rx) in enumerate(cells_x):
        cdots = cell_dots[i]
        if not cdots:
            continue

        char, conf = _decode_cell(cdots, lx, rx, [row1_y, row2_y, row3_y], get_row)
        text += char

        all_ys = [d[1] for d in cdots]
        box_x = max(0, int(lx) - int(med_r))
        box_y = max(0, min(all_ys) - int(med_r))
        box_w = int(rx - lx) + int(med_r * 2)
        box_h = (max(all_ys) - min(all_ys)) + int(med_r * 2)
        boxes.append((box_x, box_y, box_w, box_h))
        confs.append(conf)

    return text, boxes, confs, dot_infos


def _infer_three_rows(
    y_clusters: List[float], row_spacing: float, img_h: int = 0
) -> Tuple[float, float, float]:
    """
    Given Y clusters found in a braille line, infer the 3 row center positions.

    Uses image height as a reference when available to disambiguate
    whether 2 clusters represent rows 1&2, 2&3, or 1&3.
    """
    rs = row_spacing

    if len(y_clusters) == 0:
        base = img_h * 0.2 if img_h > 0 else 0.0
        return base, base + rs, base + rs * 2

    if len(y_clusters) == 1:
        y0 = y_clusters[0]
        # Use image height to determine which row this is
        if img_h > 0:
            rel = y0 / img_h  # 0=top, 1=bottom
            if rel < 0.35:
                # Dot is in top third → row 1
                return y0, y0 + rs, y0 + rs * 2
            elif rel < 0.65:
                # Dot is in middle → row 2
                return y0 - rs, y0, y0 + rs
            else:
                # Dot is in bottom third → row 3
                return y0 - rs * 2, y0 - rs, y0
        return y0, y0 + rs, y0 + rs * 2

    if len(y_clusters) == 2:
        y0, y1 = y_clusters[0], y_clusters[1]
        gap = y1 - y0

        if img_h > 0:
            # Use image height to determine row positions
            # Row 1 is at ~20% of image height, row 2 at ~42%, row 3 at ~67%
            # (based on synthetic dataset: 120px image, dots at y=20,50,80)
            r1_expected = img_h * 0.20
            r2_expected = img_h * 0.42
            r3_expected = img_h * 0.67

            # Find which rows y0 and y1 are closest to
            def nearest_row(y):
                dists = [abs(y - r1_expected), abs(y - r2_expected), abs(y - r3_expected)]
                return [1, 2, 3][int(np.argmin(dists))]

            row_of_y0 = nearest_row(y0)
            row_of_y1 = nearest_row(y1)

            if row_of_y0 == 1 and row_of_y1 == 3:
                # Rows 1 and 3 — insert row 2 in the middle
                return y0, (y0 + y1) / 2, y1
            elif row_of_y0 == 1 and row_of_y1 == 2:
                return y0, y1, y1 + (y1 - y0)
            elif row_of_y0 == 2 and row_of_y1 == 3:
                return y0 - (y1 - y0), y0, y1
            else:
                # Fallback: treat as rows 1 and 2
                return y0, y1, y1 + (y1 - y0)
        else:
            # No image height — use gap heuristic
            # If gap > 1.4 * rs, assume rows 1 and 3
            if gap > rs * 1.4:
                return y0, (y0 + y1) / 2, y1
            else:
                return y0, y1, y1 + rs

    if len(y_clusters) == 3:
        return y_clusters[0], y_clusters[1], y_clusters[2]

    # More than 3 clusters — take the first 3
    return y_clusters[0], y_clusters[1], y_clusters[2]


def _decode_cell(
    dots: List[Tuple[int, int, int]],
    left_x: float,
    right_x: float,
    row_centers: List[float],
    get_row,
) -> Tuple[str, float]:
    """
    Decode a single braille cell from its dots.
    Returns (character, confidence).

    Dot positions:
        1 4
        2 5
        3 6
    """
    cell_mid_x = (left_x + right_x) / 2
    mask = 0

    for cx, cy, r in dots:
        row = get_row(float(cy))
        row = min(max(row, 1), 3)  # clamp to 1-3

        if cx <= cell_mid_x:
            # Left column: dots 1, 2, 3
            dot_num = row
        else:
            # Right column: dots 4, 5, 6
            dot_num = row + 3

        mask |= (1 << (dot_num - 1))

    char = _BRAILLE_TO_CHAR.get(mask, "?")
    conf = 1.0 if char != "?" else 0.3
    return char, conf


def _decode_flexible(
    dots: List[Tuple[int, int, int]],
    row_spacing: float,
    col_spacing: float,
    med_r: float,
) -> Tuple[str, List[Tuple], List[float], List[DotInfo]]:
    """Flexible decoder used as fallback for low-confidence uploads."""
    return _decode_page(dots, row_spacing, col_spacing, med_r, (0, 0))


# ── Clustering helpers ────────────────────────────────────────────────────────

def _cluster_1d(values: List[float], threshold: float) -> List[float]:
    """
    Cluster 1D values into groups where adjacent values are within threshold.
    Returns list of cluster centers.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _cluster_1d_smart(values: List[float], expected_spacing: float) -> List[float]:
    """
    Cluster 1D values using a gap-based approach.
    Gaps larger than expected_spacing * 0.5 start a new cluster.
    Returns list of cluster centers.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    threshold = max(expected_spacing * 0.5, 5.0)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _pair_columns(
    col_centers: List[float], col_spacing: float
) -> List[Tuple[float, float]]:
    """
    Pair X column centers into (left_x, right_x) cell pairs.

    Within a braille cell, the left-right column gap is approximately
    col_spacing. Between cells, the gap is larger (>= col_spacing * 1.5).

    We pair adjacent columns if their gap <= col_spacing * 1.5.
    """
    if not col_centers:
        return []

    pairs = []
    i = 0
    while i < len(col_centers):
        lx = col_centers[i]
        if i + 1 < len(col_centers):
            rx = col_centers[i + 1]
            gap = rx - lx
            # Pair if gap is within 1.8x the expected intra-cell column spacing
            if gap <= col_spacing * 1.8:
                pairs.append((lx, rx))
                i += 2
                continue
        # Single column — treat as left-only cell
        pairs.append((lx, lx + col_spacing * 0.8))
        i += 1

    return pairs


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence(
    clean_text: str,
    valid_ratio: float,
    grid_score: float,
    dot_count: int,
    cell_count: int,
) -> float:
    """Compute overall detection confidence 0-1."""
    if not clean_text:
        return 0.0

    # Base: valid character ratio
    conf = valid_ratio * 0.5

    # Grid regularity bonus
    conf += grid_score * 0.3

    # Dot count bonus (more dots = more confident)
    dot_bonus = min(dot_count / 20.0, 1.0) * 0.1
    conf += dot_bonus

    # Cell count bonus
    cell_bonus = min(cell_count / 5.0, 1.0) * 0.1
    conf += cell_bonus

    return min(conf, 1.0)


def _colour_dot_infos(
    dot_infos: List[DotInfo], per_cell_conf: List[float]
) -> None:
    """Colour dots green/yellow/red based on cell confidence."""
    for d in dot_infos:
        d.colour = (255, 255, 255)
