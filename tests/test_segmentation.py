"""Tissue-mask checks on drawn images, with no whole-slide file."""

import numpy as np

from src.dataset import SATURATION_CEILING, SATURATION_FLOOR, segment_tissue


def test_blank_glass_is_not_tissue():
    rng = np.random.default_rng(0)
    glass = np.full((360, 420, 3), 242, dtype=np.uint8)
    glass = np.clip(glass.astype(np.int16) + rng.integers(-2, 3, size=glass.shape), 0, 255)
    mask, _ = segment_tissue(glass.astype(np.uint8))
    assert mask.mean() == 0


def test_stained_block_is_kept_and_glass_margin_is_dropped():
    image = np.full((400, 400, 3), 242, dtype=np.uint8)
    image[60:340, 70:330] = (190, 110, 170)
    image[80:320:6, 90:310:6] = (60, 24, 96)
    mask, threshold = segment_tissue(image)
    assert SATURATION_FLOOR <= threshold <= SATURATION_CEILING
    assert mask[200, 200] == 255
    assert mask[5, 5] == 0
    assert mask[390, 390] == 0
