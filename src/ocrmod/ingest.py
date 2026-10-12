"""Step 1: render PDF pages to images, then fix rotation and skew.

Skew is found with a projection profile: rotate a small binarised copy through a range
of angles and keep the angle where the row sums are "spikiest" (text lines line up with pixel rows).
Quarter turns are found by checking where each character's nearest neighbour sits:
beside it for upright text, above or below it for sideways text. OpenCV alone cannot tell an
upside-down page from an upright one, so 180 uses PaddleOCR's orientation classifier
when it is installed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import pymupdf
import numpy as np

from .schemas import IngestedDoc

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_pages(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    """Render each page to a BGR image."""
    images = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return images


def _binarise_small(img: np.ndarray, width: int = 1000) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = width / gray.shape[1]
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _profile_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return float(np.var(rotated.sum(axis=1, dtype=np.float64)))


def estimate_skew(img: np.ndarray, max_degrees: float = 5.0, step: float = 0.2) -> tuple[float, float]:
    """Return (angle to rotate by to straighten, profile score at that angle)."""
    binary = _binarise_small(img)
    if binary.sum() == 0:
        return 0.0, 0.0
    angles = np.arange(-max_degrees, max_degrees + step / 2, step)
    scores = [_profile_score(binary, a) for a in angles]
    best = int(np.argmax(scores))
    return float(angles[best]), scores[best]


def is_sideways(img: np.ndarray, max_chars: int = 1200) -> bool:
    """True when the text lines run vertically (page needs a quarter turn).

    In normal text a character's nearest neighbour sits beside it on the same line,
    not on the line above or below, so we vote on the direction to each nearest neighbour.
    """
    binary = _binarise_small(img)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    w, h, area = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    keep = (area >= 4) & (w < 40) & (h < 40)  # character-sized; drops ruling lines, specks
    pts = centroids[1:][keep]
    if len(pts) < 20:
        return False
    if len(pts) > max_chars:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_chars, replace=False)]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(dist, np.inf)
    nearest = d[np.arange(len(pts)), dist.argmin(axis=1)]
    horizontal = np.sum(np.abs(nearest[:, 0]) > np.abs(nearest[:, 1]))
    return (len(pts) - horizontal) > 1.5 * horizontal


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by `angle` degrees, expanding the canvas, white fill."""
    if angle % 360 == 0:
        return img
    for k, a in ((1, 90), (2, 180), (3, 270)):
        if abs(angle % 360 - a) < 1e-6:
            return np.ascontiguousarray(np.rot90(img, k))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


class _PaddleOrientation:
    """Optional 0/90/180/270 classifier (PP-LCNet doc orientation)."""

    def __init__(self):
        from paddleocr import DocImgOrientationClassification  # type: ignore

        self.model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")

    def angle(self, img: np.ndarray) -> float:
        res = next(iter(self.model.predict(img)))
        label = res["label_names"][0]
        return float(label)


def _paddle_orientation(enabled: bool) -> _PaddleOrientation | None:
    if not enabled:
        return None
    try:
        return _PaddleOrientation()
    except Exception as exc:  # paddleocr missing or no model download yet
        log.info("Paddle orientation classifier unavailable (%s); using OpenCV only", exc)
        return None


def straighten(img: np.ndarray, max_skew: float, classifier=None) -> tuple[np.ndarray, float]:
    """Fix quarter-turn rotation and small skew. Returns (image, total degrees applied)."""
    applied = 0.0
    if classifier is not None:
        # Classifier reports how far the page is rotated; undo it.
        detected = classifier.angle(img)
        if detected:
            img = rotate(img, detected)
            applied += detected
    elif is_sideways(img):
        # Without a classifier we can't tell 90 from 270; 90 is a guess, upside-down
        # results still read fine for the OCR engines that handle 180 internally.
        img = rotate(img, 90)
        applied += 90
    skew, _ = estimate_skew(img, max_skew)
    if abs(skew) >= 0.2:
        img = rotate(img, skew)
        applied += skew
    return img, applied


def ingest(pdf_path: Path, out_dir: Path, dpi: int = 250, max_skew: float = 5.0,
           use_paddle_orientation: bool = True) -> IngestedDoc:
    """Render, straighten, and save page PNGs to out_dir/pages/."""
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    classifier = _paddle_orientation(use_paddle_orientation)
    paths, rotations = [], []
    for i, img in enumerate(render_pages(pdf_path, dpi)):
        fixed, applied = straighten(img, max_skew, classifier)
        p = pages_dir / f"page-{i + 1:03d}.png"
        cv2.imwrite(str(p), fixed)
        paths.append(p)
        rotations.append(round(applied, 2))
    return IngestedDoc(pdf_path=pdf_path, doc_id=file_sha256(pdf_path), page_images=paths,
                       dpi=dpi, rotations=rotations)
