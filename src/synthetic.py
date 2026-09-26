"""Build a small fake H&E slide so the pipeline can run without a biopsy file.

The picture is not tissue. It is a four-region cartoon with a glass margin:

- dense purple "nuclei" (hypercellular-looking)
- sparse nuclei on a pink background (stroma-looking)
- pale, almost structureless pink-gray (necrosis-looking)
- redder blobs (hemorrhage-looking)
- a mixed disk in the center

Real hematoxylin and eosin morphology is far more subtle. The cartoon only
exists so tiling, embedding, clustering, and the heatmap can be exercised
on a laptop. Cluster names above are visual hints, not labels the model knows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def render_synthetic_he(size: int = 3072, seed: int = 0) -> np.ndarray:
    """Return an RGB uint8 canvas with a glass border and four stained regions."""

    if size < 1024 or size % 256 != 0:
        raise ValueError("size must be a multiple of 256 and at least 1024")

    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 242, dtype=np.uint8)
    glass_noise = rng.normal(0, 1.5, size=image.shape)
    image = np.clip(image.astype(np.float32) + glass_noise, 0, 255).astype(np.uint8)

    margin = 256
    inner = size - 2 * margin
    mid = margin + inner // 2
    # Quadrants of the stained field, then a disk that mixes two of them.
    regions = [
        ((margin, mid, margin, mid), (214, 156, 186), (72, 36, 112), 0.18, 3),
        ((margin, mid, mid, size - margin), (232, 186, 196), (96, 48, 92), 0.025, 3),
        ((mid, size - margin, margin, mid), (214, 198, 176), (150, 140, 120), 0.012, 2),
        ((mid, size - margin, mid, size - margin), (196, 92, 98), (120, 24, 36), 0.08, 5),
    ]
    for (y0, y1, x0, x1), background, nucleus, density, nucleus_px in regions:
        _paint_cellular_region(
            image,
            y0,
            y1,
            x0,
            x1,
            background=background,
            nucleus=nucleus,
            density=density,
            nucleus_px=nucleus_px,
            rng=rng,
        )

    center = size // 2
    radius = inner // 8
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    _paint_cellular_region(
        image,
        center - radius,
        center + radius,
        center - radius,
        center + radius,
        background=(200, 140, 150),
        nucleus=(70, 40, 100),
        density=0.09,
        nucleus_px=3,
        rng=rng,
        keep=disk,
    )
    return image


def generate_synthetic_wsi(output_path: str | Path, size: int = 3072, seed: int = 0) -> Path:
    """Write a pyramidal tiled TIFF that OpenSlide can open.

    libvips (``pyvips``) is preferred because OpenSlide reliably reads its
    JPEG pyramids. If libvips is not installed, a single-resolution tiled
    TIFF is written with ``tifffile`` instead. Either file is a stand-in for
    an ``.svs`` biopsy, already at the 20x pixel size the tiler expects when
    a scan does not record an objective power.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = render_synthetic_he(size=size, seed=seed)
    if _write_with_pyvips(rgb, output_path):
        logger.info("Wrote synthetic pyramidal slide to %s", output_path)
        return output_path
    _write_with_tifffile(rgb, output_path)
    logger.info("Wrote synthetic single-level slide to %s", output_path)
    return output_path


def _paint_cellular_region(
    image: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    background: tuple[int, int, int],
    nucleus: tuple[int, int, int],
    density: float,
    nucleus_px: int,
    rng: np.random.Generator,
    keep: np.ndarray | None = None,
) -> None:
    """Fill a rectangle with a flat stain color and speckled darker nuclei."""

    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = min(image.shape[0], int(y1))
    x1 = min(image.shape[1], int(x1))
    height, width = y1 - y0, x1 - x0
    if height <= 0 or width <= 0:
        return

    region = np.empty((height, width, 3), dtype=np.uint8)
    region[:] = background
    speckles = rng.random((height, width)) < density
    kernel_size = max(1, int(nucleus_px))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        speckles = cv2.dilate(speckles.astype(np.uint8) * 255, kernel) > 0
    region[speckles] = nucleus
    jitter = rng.normal(0, 4, size=region.shape)
    region = np.clip(region.astype(np.float32) + jitter, 0, 255).astype(np.uint8)

    target = image[y0:y1, x0:x1]
    if keep is None:
        target[:] = region
        return
    # Assign through the view. Chained `image[slice][mask] = ...` writes to a
    # temporary copy and would silently drop the disk.
    target[keep[y0:y1, x0:x1]] = region[keep[y0:y1, x0:x1]]


def _write_with_pyvips(rgb: np.ndarray, output_path: Path) -> bool:
    try:
        import pyvips
    except ImportError:
        logger.warning("pyvips is not installed; falling back to a flat TIFF")
        return False

    array = np.ascontiguousarray(rgb)
    height, width, bands = array.shape
    # Keep the byte buffer alive until copy() has its own pixel storage.
    buffer = array.tobytes()
    image = pyvips.Image.new_from_memory(buffer, width, height, bands, "uchar")
    image = image.copy()
    try:
        image.tiffsave(
            str(output_path),
            tile=True,
            pyramid=True,
            compression="jpeg",
            Q=90,
            bigtiff=True,
            tile_width=256,
            tile_height=256,
        )
    except Exception as exc:  # libvips reports writer errors as a broad exception type
        logger.warning("libvips could not write %s (%s). Falling back to tifffile.", output_path, exc)
        return False
    return True


def _write_with_tifffile(rgb: np.ndarray, output_path: Path) -> None:
    import tifffile

    tifffile.imwrite(
        output_path,
        np.ascontiguousarray(rgb),
        photometric="rgb",
        tile=(256, 256),
        compression="jpeg",
        compressionargs={"level": 90},
    )
