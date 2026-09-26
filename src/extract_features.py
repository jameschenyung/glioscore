"""Embed tissue patches with a pretrained vision network.

Each 256x256 patch is resized to the input size the network was trained
with (224x224 for ResNet50 and DINOv2-ViT-S/14) and normalized with
ImageNet mean and standard deviation. The classification layer is removed,
so the output is one feature vector per patch.

Row ``i`` of the saved matrix is the embedding of row ``i`` in the tiling
manifest. Do not shuffle the loader: the cluster heatmap zips embeddings
back to ``(x, y)`` by that shared order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Both supported encoders consume 224x224. DINOv2 splits the image into 14x14
# patches, and 224 is divisible by 14. ResNet50 was trained at this size.
MODEL_INPUT_SIZE = 224


class PatchDataset(Dataset):
    """RGB patches listed in a tiling manifest.

    Args:
        manifest: DataFrame with a ``path`` column pointing at patch images.
            Any extra columns (coordinates, tissue fraction) are ignored here
            and carried along by the caller.
        transform: Torchvision transform that returns a float tensor.
    """

    def __init__(self, manifest: pd.DataFrame, transform) -> None:
        if "path" not in manifest.columns:
            raise ValueError("manifest must contain a 'path' column")
        if len(manifest) == 0:
            raise ValueError("manifest is empty; there are no patches to embed")
        self.paths = [str(p) for p in manifest["path"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor


class FeatureEncoder(nn.Module):
    """Wrap a backbone so ``forward`` always returns an (N, D) embedding."""

    def __init__(self, backbone: nn.Module, feature_dim: int, name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.name = name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, dict):
            # Some self-supervised heads return a dict of CLS and patch tokens.
            for key in ("x_norm_clstoken", "x_norm_patchtokens", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
        if isinstance(features, (tuple, list)):
            features = features[0]
        # Some backbones return a feature map (N, C, H, W) instead of a vector.
        if features.ndim > 2:
            features = nn.functional.adaptive_avg_pool2d(features, output_size=1)
            features = torch.flatten(features, 1)
        return features


def get_device(preference: str = "auto") -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU.

    ``preference`` may be ``auto``, ``cpu``, ``cuda``, or ``mps``.
    """

    preference = preference.lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available on this machine")
        return torch.device("mps")
    if preference != "auto":
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int = MODEL_INPUT_SIZE):
    """Resize and ImageNet-normalize a PIL image into a model tensor."""

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_encoder(model_name: str = "resnet50", weights: str = "imagenet") -> FeatureEncoder:
    """Construct a pretrained embedding network.

    Args:
        model_name: ``resnet50`` (torchvision, 2048-d) or ``dinov2_vits14``
            (torch.hub, 384-d). ResNet50 is the default because its weights
            ship with torchvision and do not require a separate hub checkout.
        weights: ``imagenet`` loads the published checkpoint. ``none`` leaves
            the network randomly initialized, which is only useful for tests.
    """

    key = model_name.lower()
    if weights not in {"imagenet", "none"}:
        raise ValueError("weights must be 'imagenet' or 'none'")

    if key == "resnet50":
        pretrained = ResNet50_Weights.IMAGENET1K_V2 if weights == "imagenet" else None
        backbone = resnet50(weights=pretrained)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return FeatureEncoder(backbone, feature_dim=feature_dim, name="resnet50")

    if key == "dinov2_vits14":
        if weights != "imagenet":
            raise ValueError("dinov2_vits14 only supports weights='imagenet'")
        logger.info("Loading dinov2_vits14 from torch.hub (first run downloads weights)")
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        return FeatureEncoder(backbone, feature_dim=384, name="dinov2_vits14")

    raise ValueError("model_name must be 'resnet50' or 'dinov2_vits14'")


def extract_embeddings(
    manifest: pd.DataFrame,
    model_name: str = "resnet50",
    weights: str = "imagenet",
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
) -> tuple[np.ndarray, torch.device, str]:
    """Run every manifest patch through the encoder.

    Returns:
        embeddings: float32 array shaped (N_patches, feature_dim), aligned
            with ``manifest`` row order.
        device: The torch device that actually ran the model.
        encoder_name: Canonical model name stored in the report.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    torch_device = get_device(device)
    encoder = build_encoder(model_name, weights=weights)
    encoder.eval()
    encoder.to(torch_device)
    dataset = PatchDataset(manifest, build_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    logger.info(
        "Extracting %s embeddings for %s patches on %s (batch size %s)",
        encoder.name,
        len(dataset),
        torch_device,
        batch_size,
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="embed patches", unit="batch"):
            batch = batch.to(torch_device, non_blocking=torch_device.type == "cuda")
            features = encoder(batch)
            chunks.append(features.detach().float().cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(manifest):
        raise RuntimeError(
            f"Expected {len(manifest)} embeddings, got {embeddings.shape[0]}"
        )
    return embeddings, torch_device, encoder.name


def save_embeddings(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    output_dir: str | Path,
    fmt: str = "npy",
) -> Path:
    """Write the embedding matrix and the coordinate index beside it.

    Args:
        embeddings: Array shaped (N_patches, feature_dim).
        manifest: Tiling manifest aligned row-for-row with ``embeddings``.
        output_dir: Directory that already holds the manifest.
        fmt: ``npy`` writes ``embeddings.npy``. ``parquet`` writes
            ``embeddings.parquet`` (coordinates plus one column per feature).
            ``both`` writes the two files.

    Returns:
        Path of the primary matrix file (``.npy``, or ``.parquet`` when
        ``fmt`` is ``parquet``).
    """

    if fmt not in {"npy", "parquet", "both"}:
        raise ValueError("fmt must be 'npy', 'parquet', or 'both'")
    if len(embeddings) != len(manifest):
        raise ValueError("embeddings and manifest must have the same number of rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_columns = [c for c in ("patch_id", "x", "y", "tissue_fraction") if c in manifest.columns]
    index = manifest[index_columns].reset_index(drop=True)
    index.to_parquet(output_dir / "patch_index.parquet", index=False)

    npy_path = output_dir / "embeddings.npy"
    parquet_path = output_dir / "embeddings.parquet"
    if fmt in {"npy", "both"}:
        np.save(npy_path, embeddings)
    if fmt in {"parquet", "both"}:
        feature_frame = pd.DataFrame(
            embeddings,
            columns=[f"f{i:04d}" for i in range(embeddings.shape[1])],
        )
        pd.concat([index, feature_frame], axis=1).to_parquet(parquet_path, index=False)

    if fmt == "parquet":
        return parquet_path
    return npy_path
