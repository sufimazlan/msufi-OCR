import cv2
import numpy as np
import pytest

from ocrmod.ingest import estimate_skew, rotate, straighten


def text_page(w=1200, h=1600) -> np.ndarray:
    img = np.full((h, w, 3), 255, np.uint8)
    for i, y in enumerate(range(120, h - 120, 45)):
        cv2.putText(img, f"Line {i} invoice total amount RM 1,234.50 due", (80, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


@pytest.mark.parametrize("skew", [-3.0, 1.5, 4.0])
def test_estimate_skew_undoes_rotation(skew):
    angle, _ = estimate_skew(rotate(text_page(), skew))
    assert angle == pytest.approx(-skew, abs=0.4)


def test_straighten_fixes_quarter_turn():
    page = text_page()
    fixed, applied = straighten(rotate(page, 90), max_skew=5.0, classifier=None)
    assert fixed.shape[:2] == page.shape[:2]
    assert applied % 180 == pytest.approx(90, abs=0.5)


def test_straight_page_untouched():
    page = text_page()
    fixed, applied = straighten(page, max_skew=5.0, classifier=None)
    assert applied == 0
    assert fixed.shape == page.shape
