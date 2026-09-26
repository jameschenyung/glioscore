"""Load whole-slide images, isolate tissue, and cut fixed-size patches.

A whole-slide image (WSI) is a pyramid of one glass microscope slide. The
finest level (level 0) can be tens of thousands of pixels on a side, and most
of those pixels are empty glass. Downstream models never see the raw slide.
They see small squares ("patches" or "tiles") taken only where there is
stained tissue.

Coordinate convention used everywhere in this package
-----------------------------------------------------
``x`` and ``y`` are the top-left corner of a patch in **level-0 pixels**.
OpenSlide's ``read_region`` always expects locations in that same level-0
frame, even when the pixels are read from a coarser pyramid level. Keeping
one coordinate system makes the cluster heatmap line up with the slide.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Conventional microns-per-pixel of a 20x brightfield scan. Used only when a
# slide stores resolution as MPP and not as an objective power.
MPP_AT_20X = 0.5

# Saturation below this (0-255) is treated as unstained glass. A two-spike
# histogram can make Otsu report 0, which would keep every faintly colored pixel.
SATURATION_FLOOR = 12

# Otsu on a slide that is mostly glass, plus a few strongly stained regions,
# often lands far above the saturation of light eosin. Cap it so pale stained
# tissue is not thrown away with the glass. Dense nuclei still pass either way.
SATURATION_CEILING = 40

# Pixels brighter than this in grayscale are glass or glare, not stain.
GLASS_GRAY_MIN = 230


class OpenSlideUnavailableError(ImportError):
    """Raised when the OpenSlide Python bindings or system library are missing."""


class NoTissueFoundError(RuntimeError):
    """Raised when segmentation keeps no patch above the tissue-fraction cutoff."""


@dataclass(frozen=True)
class SlideGeometry:
    """Magnification math for one slide.

    Attributes:
        objective_power: Stated scan power at level 0 (for example 40.0), or
            None when the file does not record it.
        target_magnification: Power the patches should represent (default 20).
        target_downsample: How many level-0 pixels span one pixel at the
            target power. A 40x scan read at 20x has downsample 2.
        patch_size: Width and height of each saved patch, in pixels.
        patch_size_level0: Width of that patch measured in level-0 pixels.
        stride_level0: Grid step in level-0 pixels.
    """

    objective_power: float | None
    target_magnification: float
    target_downsample: float
    patch_size: int
    patch_size_level0: int
    stride_level0: int


def load_wsi(slide_path: str | Path):
    """Open a ``.svs``, ``.tif``, or other OpenSlide-supported whole-slide file.

    The returned object is an ``openslide.OpenSlide`` handle. Callers must
    close it (``slide.close()``) when they are done reading.
    """

    try:
        import openslide
    except ImportError as exc:
        raise OpenSlideUnavailableError(
            "openslide-python is not installed, or the OpenSlide system library "
            "is missing. See the README section 'System dependency: OpenSlide'."
        ) from exc

    path = Path(slide_path)
    if not path.is_file():
        raise FileNotFoundError(f"Slide not found: {path}")

    try:
        slide = openslide.OpenSlide(str(path))
    except openslide.OpenSlideError as exc:
        raise RuntimeError(
            f"OpenSlide could not read {path.name}. The file may be corrupt or "
            "use a format this OpenSlide build does not support."
        ) from exc

    logger.info(
        "Opened %s | levels=%s | level-0 size=%s",
        path.name,
        slide.level_count,
        slide.dimensions,
    )
    return slide


def infer_objective_power(slide) -> float | None:
    """Best-effort scan magnification at level 0.

    Vendors store this differently. Aperio ``.svs`` files usually set
    ``openslide.objective-power``. Others only store microns per pixel.
    When neither is present we return None and the tiler treats level 0 as
    already being at the requested magnification.
    """

    raw_power = slide.properties.get("openslide.objective-power")
    if raw_power:
        try:
            power = float(raw_power)
        except ValueError:
            power = 0.0
        if power > 0:
            return power

    raw_mpp = slide.properties.get("openslide.mpp-x")
    if raw_mpp:
        try:
            mpp = float(raw_mpp)
        except ValueError:
            mpp = 0.0
        if mpp > 0:
            # 20x is ~0.5 µm/px and 40x is ~0.25 µm/px, so power ≈ 10 / mpp.
            return 10.0 / mpp
    return None


def slide_geometry(
    slide,
    patch_size: int = 256,
    target_magnification: float = 20.0,
    stride: int | None = None,
) -> SlideGeometry:
    """Translate a requested magnification into level-0 pixel sizes."""

    if patch_size < 16:
        raise ValueError("patch_size must be at least 16 pixels")
    if target_magnification <= 0:
        raise ValueError("target_magnification must be positive")

    objective = infer_objective_power(slide)
    if objective is None:
        logger.warning(
            "Slide has no objective power or MPP. Assuming level 0 is already %.1fx.",
            target_magnification,
        )
        target_downsample = 1.0
    else:
        # Example: objective 40, target 20 -> each output pixel covers 2 level-0 pixels.
        target_downsample = objective / target_magnification
        if target_downsample < 1:
            logger.warning(
                "Requested %.1fx but the scan is only %.1fx. Reading level 0 "
                "without upscaling.",
                target_magnification,
                objective,
            )
            target_downsample = 1.0

    stride_px = patch_size if stride is None else stride
    if stride_px < 1:
        raise ValueError("stride must be at least 1 pixel")

    return SlideGeometry(
        objective_power=objective,
        target_magnification=target_magnification,
        target_downsample=target_downsample,
        patch_size=patch_size,
        patch_size_level0=max(1, int(round(patch_size * target_downsample))),
        stride_level0=max(1, int(round(stride_px * target_downsample))),
    )


def segment_tissue(
    thumbnail_rgb: np.ndarray,
    saturation_floor: int = SATURATION_FLOOR,
    saturation_ceiling: int = SATURATION_CEILING,
    glass_gray_min: int = GLASS_GRAY_MIN,
) -> tuple[np.ndarray, float]:
    """Build a binary tissue mask from a low-resolution RGB thumbnail.

    Hematoxylin and eosin (the routine biopsy stain) colors nuclei blue-purple
    and cytoplasm pink. Empty glass is nearly white, so it has very low color
    saturation in HSV. Otsu's method picks a saturation threshold that splits
    "has stain" from "does not". That raw cut is then clipped to
    ``[saturation_floor, saturation_ceiling]``: the floor stops a degenerate
    threshold of 0 from calling JPEG noise tissue, and the ceiling stops a
    glass-dominated histogram from setting the cut so high that light eosin
    disappears. A brightness gate then drops glare. Small specks are removed.

    Args:
        thumbnail_rgb: uint8 array shaped (H, W, 3) in RGB order.
        saturation_floor: Lowest HSV saturation that may be called tissue.
        saturation_ceiling: Highest cut Otsu is allowed to apply.
        glass_gray_min: Grayscale values at or above this are background.

    Returns:
        mask: uint8 array shaped (H, W), 255 on tissue and 0 on glass.
        applied_threshold: Saturation cut actually used, after clipping Otsu.
    """

    if thumbnail_rgb.ndim != 3 or thumbnail_rgb.shape[2] != 3:
        raise ValueError("thumbnail_rgb must have shape (H, W, 3)")
    if thumbnail_rgb.dtype != np.uint8:
        thumbnail_rgb = np.clip(thumbnail_rgb, 0, 255).astype(np.uint8)

    if saturation_ceiling < saturation_floor:
        raise ValueError("saturation_ceiling must be >= saturation_floor")

    hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
    # Median blur knocks out single-pixel compression noise before thresholding.
    saturation = cv2.medianBlur(hsv[:, :, 1], 5)
    raw_otsu, _ = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    applied_threshold = float(np.clip(raw_otsu, saturation_floor, saturation_ceiling))
    _, sat_mask = cv2.threshold(
        saturation, applied_threshold, 255, cv2.THRESH_BINARY
    )
    logger.debug(
        "Saturation Otsu raw=%.1f applied=%.1f (clip [%s, %s])",
        raw_otsu,
        applied_threshold,
        saturation_floor,
        saturation_ceiling,
    )

    gray = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2GRAY)
    # BINARY_INV keeps the darker-than-glass pixels (stained tissue).
    _, dark_mask = cv2.threshold(gray, glass_gray_min, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(sat_mask, dark_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = _drop_small_components(mask, min_area_fraction=0.0005)
    return mask, applied_threshold


def _drop_small_components(mask: np.ndarray, min_area_fraction: float) -> np.ndarray:
    """Delete connected tissue blobs smaller than a fraction of the thumbnail."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    # A pixel floor keeps dust out without deleting small biopsy fragments.
    min_area = max(32, int(round(min_area_fraction * mask.size)))
    cleaned = np.zeros_like(mask)
    for label_id in range(1, count):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def read_thumbnail(slide, max_side: int = 2048) -> Image.Image:
    """RGB thumbnail whose longest side is about ``max_side`` pixels."""

    width, height = slide.dimensions
    scale = max_side / max(width, height)
    thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return slide.get_thumbnail(thumb_size).convert("RGB")


def read_patch(
    slide,
    x: int,
    y: int,
    patch_size_level0: int,
    out_size: int,
) -> Image.Image:
    """Read one patch and resize it to ``out_size`` x ``out_size`` RGB.

    ``x`` and ``y`` are level-0 coordinates. The read itself may come from a
    coarser pyramid level so a 20x patch on a 40x scan does not pull extra
    pixels and then throw them away.
    """

    desired_downsample = patch_size_level0 / float(out_size)
    level = slide.get_best_level_for_downsample(desired_downsample)
    level_downsample = float(slide.level_downsamples[level])
    read_size = max(1, int(round(patch_size_level0 / level_downsample)))
    # Rounding can ask for one pixel past a coarse pyramid level. Clamp to it.
    level_width, level_height = slide.level_dimensions[level]
    origin_x = int(np.floor(x / level_downsample))
    origin_y = int(np.floor(y / level_downsample))
    read_size = min(read_size, level_width - origin_x, level_height - origin_y)
    read_size = max(1, read_size)
    # read_region origin is always level 0, even when `level` is not.
    region = slide.read_region((int(x), int(y)), level, (read_size, read_size))
    rgb = region.convert("RGB")
    if rgb.size != (out_size, out_size):
        rgb = rgb.resize((out_size, out_size), Image.Resampling.BILINEAR)
    return rgb


def _tissue_fraction(
    mask: np.ndarray,
    x: int,
    y: int,
    patch_size_level0: int,
    slide_width: int,
    slide_height: int,
) -> float:
    """Fraction of thumbnail-mask pixels inside this level-0 patch that are tissue."""

    mask_h, mask_w = mask.shape[:2]
    x0 = int(np.floor(x * mask_w / slide_width))
    y0 = int(np.floor(y * mask_h / slide_height))
    x1 = int(np.ceil((x + patch_size_level0) * mask_w / slide_width))
    y1 = int(np.ceil((y + patch_size_level0) * mask_h / slide_height))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask_w, max(x0 + 1, x1)), min(mask_h, max(y0 + 1, y1))
    window = mask[y0:y1, x0:x1]
    if window.size == 0:
        return 0.0
    return float((window > 0).mean())


def extract_patches(
    slide_path: str | Path,
    output_dir: str | Path,
    patch_size: int = 256,
    target_magnification: float = 20.0,
    stride: int | None = None,
    tissue_threshold: float = 0.5,
    max_patches: int | None = None,
    patch_format: str = "png",
) -> pd.DataFrame:
    """Tile stained tissue into patches and write a manifest.

    Steps:
        1. Open the slide and choose the level-0 patch size that corresponds
           to ``patch_size`` pixels at ``target_magnification`` (default 20x).
        2. Segment tissue on a thumbnail with HSV Otsu thresholding.
        3. Walk a regular grid. Keep a tile when the share of mask pixels
           inside it is at least ``tissue_threshold``.
        4. Save each kept patch and a row of coordinates.

    The manifest columns are ``patch_id``, ``x``, ``y``, ``tissue_fraction``,
    and ``path``. ``x`` and ``y`` are level-0 top-left coordinates. Rows are
    ordered row-major (y, then x), which is also the row order of the
    embedding matrix written later.

    Returns:
        The manifest as a DataFrame. The same table is written to
        ``output_dir/manifest.parquet`` and ``manifest.csv``.
    """

    if not 0.0 < tissue_threshold <= 1.0:
        raise ValueError("tissue_threshold must be in the interval (0, 1]")
    if patch_format not in {"png", "jpg"}:
        raise ValueError("patch_format must be 'png' or 'jpg'")

    output_dir = Path(output_dir)
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    slide = load_wsi(slide_path)
    try:
        geometry = slide_geometry(
            slide,
            patch_size=patch_size,
            target_magnification=target_magnification,
            stride=stride,
        )
        slide_width, slide_height = slide.dimensions
        thumbnail = read_thumbnail(slide)
        thumbnail_rgb = np.asarray(thumbnail)
        mask, otsu_threshold = segment_tissue(thumbnail_rgb)
        _save_tissue_overview(output_dir, thumbnail_rgb, mask)

        candidates = _candidate_tiles(
            mask=mask,
            geometry=geometry,
            slide_width=slide_width,
            slide_height=slide_height,
            tissue_threshold=tissue_threshold,
        )
        if not candidates:
            raise NoTissueFoundError(
                "No tile passed the tissue filter. The slide may be blank, the "
                "stain may be too faint, or tissue_threshold may be too high."
            )
        if max_patches is not None and len(candidates) > max_patches:
            candidates = _evenly_subsample(candidates, max_patches)
            logger.info("Subsampled tiles to max_patches=%s", max_patches)

        logger.info(
            "Saturation threshold=%.1f | kept %s patches | "
            "patch %sx%s at %.1fx (level-0 size %s, stride %s)",
            otsu_threshold,
            len(candidates),
            geometry.patch_size,
            geometry.patch_size,
            geometry.target_magnification,
            geometry.patch_size_level0,
            geometry.stride_level0,
        )

        rows: list[dict] = []
        for patch_id, (x, y, fraction) in enumerate(candidates):
            image = read_patch(
                slide,
                x=x,
                y=y,
                patch_size_level0=geometry.patch_size_level0,
                out_size=geometry.patch_size,
            )
            filename = f"{patch_id:06d}.{patch_format}"
            path = patch_dir / filename
            if patch_format == "jpg":
                image.save(path, quality=95)
            else:
                image.save(path)
            rows.append(
                {
                    "patch_id": patch_id,
                    "x": int(x),
                    "y": int(y),
                    "tissue_fraction": fraction,
                    "path": str(path.resolve()),
                }
            )
    finally:
        slide.close()

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(output_dir / "manifest.parquet", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    _write_geometry_sidecar(output_dir, geometry, slide_width, slide_height, slide_path)
    return manifest


def _candidate_tiles(
    mask: np.ndarray,
    geometry: SlideGeometry,
    slide_width: int,
    slide_height: int,
    tissue_threshold: float,
) -> list[tuple[int, int, float]]:
    """Grid locations whose tissue fraction meets the cutoff, in row-major order."""

    kept: list[tuple[int, int, float]] = []
    patch = geometry.patch_size_level0
    stride = geometry.stride_level0
    # Stop before a patch would hang off the right or bottom edge.
    y = 0
    while y + patch <= slide_height:
        x = 0
        while x + patch <= slide_width:
            fraction = _tissue_fraction(mask, x, y, patch, slide_width, slide_height)
            if fraction >= tissue_threshold:
                kept.append((x, y, fraction))
            x += stride
        y += stride
    return kept


def _evenly_subsample(
    candidates: list[tuple[int, int, float]],
    max_patches: int,
) -> list[tuple[int, int, float]]:
    """Keep ``max_patches`` tiles spread across the row-major grid order."""

    if max_patches < 1:
        raise ValueError("max_patches must be positive")
    positions = np.linspace(0, len(candidates) - 1, num=max_patches)
    indices = np.unique(np.round(positions).astype(int))
    return [candidates[int(i)] for i in indices]


def _save_tissue_overview(
    output_dir: Path,
    thumbnail_rgb: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write the thumbnail and a green overlay of the tissue mask for QA."""

    Image.fromarray(thumbnail_rgb).save(output_dir / "thumbnail.png")
    Image.fromarray(mask).save(output_dir / "tissue_mask.png")
    overlay = thumbnail_rgb.copy()
    tissue = mask > 0
    # Tint kept pixels so a student can see what the segmenter called tissue.
    overlay[tissue, 1] = np.clip(overlay[tissue, 1].astype(np.int16) + 60, 0, 255).astype(
        np.uint8
    )
    Image.fromarray(overlay).save(output_dir / "tissue_overlay.png")


def _write_geometry_sidecar(
    output_dir: Path,
    geometry: SlideGeometry,
    slide_width: int,
    slide_height: int,
    slide_path: str | Path,
) -> None:
    """Store the sizes the heatmap needs in order to use level-0 coordinates."""

    payload = {
        "slide_path": str(slide_path),
        "slide_width": slide_width,
        "slide_height": slide_height,
        "objective_power": geometry.objective_power,
        "target_magnification": geometry.target_magnification,
        "target_downsample": geometry.target_downsample,
        "patch_size": geometry.patch_size,
        "patch_size_level0": geometry.patch_size_level0,
        "stride_level0": geometry.stride_level0,
    }
    (output_dir / "geometry.json").write_text(json.dumps(payload, indent=2))
