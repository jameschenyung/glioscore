# Glioblastoma intratumoral heterogeneity pipeline

This repository measures how morphologically mixed one biopsy slide is.

A whole-slide image (WSI) is a pyramid scan of a glass slide, often several gigabytes, stored as `.svs` or `.tif`. Most of the pixels are empty glass. The pipeline:

1. Opens the slide with OpenSlide.
2. Finds stained tissue with an HSV saturation threshold (Otsu) and drops the glass.
3. Cuts the tissue into 256×256 patches at 20× magnification and records each patch's level-0 `(x, y)` coordinate.
4. Embeds every patch with a pretrained vision model (ResNet50 by default, DINOv2 ViT-S/14 optional).
5. Clusters the embeddings into morphological niches with K-Means (`k=8` by default) or a Gaussian mixture.
6. Reports one **heterogeneity score** per slide: Shannon entropy of the niche histogram, normalized by `log2(k)` so it lies in `[0, 1]`.

This is an educational research tool. It does **not** diagnose glioblastoma, grade a tumor, or name histologic structures. Niche ids are groups of patches that look similar to the model. A pathologist, or a later supervised model, would be required to say whether a niche is necrosis, microvascular proliferation, or something else.

## Score, in one paragraph

Let `p_i` be the fraction of patches assigned to niche `i`. Shannon entropy is

```
H = -sum_i p_i log2(p_i)
```

`H` is 0 when every patch shares one niche (homogeneous). It is `log2(k)` when the patches are split evenly across all `k` niches (as mixed as this clustering can measure). The printed **heterogeneity score** is `H / log2(k)`. The cluster diversity ratio is the fraction of the `k` niches that each contain at least 2% of the patches. Spatial mixing is the fraction of neighboring patches that landed in different niches: a low value means niches sit in solid regions, a high value means they are intermixed. The heatmap shows where those niches are. Color is a label, not a severity scale.

## Layout

```
main.py                         command-line runner
src/dataset.py                  OpenSlide loading, tissue mask, tiling
src/extract_features.py         patch Dataset, ResNet50 / DINOv2, embeddings
src/quantify_heterogeneity.py   K-Means / GMM, entropy, heatmap
src/synthetic.py                optional fake slide for --demo (offline fallback)
requirements.txt
```

A run writes:

```
outputs/<slide>/
  thumbnail.png                 low-resolution slide
  tissue_mask.png               white = tissue kept by the segmenter
  tissue_overlay.png            thumbnail with tissue tinted green
  patches/000000.png            256×256 RGB tiles
  manifest.csv                  patch_id, x, y, tissue_fraction, path
  geometry.json                 level-0 size and the 20× patch size
  embeddings.npy                float32 matrix, shape (N_patches, feature_dim)
  patch_index.parquet           coordinates aligned with embedding rows
  cluster_labels.npy
  cluster_heatmap.png
  cluster_proportions.png
  heterogeneity_report.json
```

`x` and `y` are the top-left corner of the patch in **level-0 pixels** (the full-resolution coordinate system OpenSlide uses). Embedding row `i` belongs to manifest row `i`.

## System dependency: OpenSlide

`openslide-python` is only a wrapper. The OpenSlide C library has to be installed first. Real `.svs` files (including the example below) need it.

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 openslide-tools
```

macOS (Homebrew):

```bash
brew install openslide
```

Windows:

Install the bundled binaries with `pip install openslide-bin` (also listed in `requirements.txt`), or download OpenSlide from [openslide.org](https://openslide.org/download/) and add the folder that contains `libopenslide-*.dll` to `PATH`.

Check the system library with:

```bash
openslide-show-properties --version 2>/dev/null || python -c "import openslide; print(openslide.__library_version__)"
```

## Install Python packages

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

On a CPU-only machine, install PyTorch from the CPU wheel index **before** the rest of the requirements. A default `pip install torch` can download several gigabytes of CUDA libraries.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you have a CUDA GPU and want the matching PyTorch build, follow the selector at [pytorch.org](https://pytorch.org) instead of the CPU index, then install the remaining requirements.

The first real run also downloads ResNet50 ImageNet weights (about 100 MB) into the torch cache.

## Run

### Example slide (recommended)

This repository does not ship patient images. The walkthrough uses a **TCGA glioblastoma (brain)** whole-slide image, `TCGA-06-0743-01Z-00-DX1.svs` (~18k×28k at 20×, ~113 MB), redistributed in the [TCGA-mini](https://huggingface.co/datasets/W8Yi/TCGA-mini) Hugging Face dataset. Users are responsible for complying with [TCGA / GDC data-use](https://gdc.cancer.gov/access-data/data-access-policies) and publication policies.

Install the Hub CLI once, then download the slide:

```bash
pip install huggingface_hub
mkdir -p data
hf download W8Yi/TCGA-mini --repo-type dataset \
  --include "slides/TCGA-06-0743-01Z-00-DX1.svs" \
  --local-dir data/tcga_gbm
```

On Windows (PowerShell):

```powershell
.\.venv\Scripts\pip install huggingface_hub
New-Item -ItemType Directory -Force -Path data | Out-Null
.\.venv\Scripts\hf download W8Yi/TCGA-mini --repo-type dataset `
  --include "slides/TCGA-06-0743-01Z-00-DX1.svs" `
  --local-dir data/tcga_gbm
```

Run the pipeline (cap patches so a full WSI finishes in a reasonable time on CPU):

```bash
python main.py \
  --slide data/tcga_gbm/slides/TCGA-06-0743-01Z-00-DX1.svs \
  --output-dir outputs/tcga_06_0743 \
  --max-patches 2000
```

With pretrained ResNet50, `k=8`, and `--max-patches 2000`, a typical run on this slide keeps on the order of **~1900** tissue patches, a **heterogeneity score around 0.65**, several large niches plus a few tiny boundary niches (diversity ratio near 0.5), and **spatial mixing** around 0.3 (mix is partly regional). Inspect `outputs/tcga_06_0743/cluster_heatmap.png`, `cluster_proportions.png`, and `heterogeneity_report.json`. Scores will move a little if you change `k`, the encoder, or the patch cap.

Your own slide:

```bash
python main.py --slide /path/to/biopsy.svs --output-dir outputs/biopsy
```

Useful knobs:

```bash
python main.py --slide biopsy.svs --output-dir outputs/biopsy \
  --magnification 20 --patch-size 256 --k 8 --method kmeans \
  --model resnet50 --device auto --max-patches 2000
```

`--model dinov2_vits14` loads DINOv2 ViT-S/14 from `torch.hub` (384-d embeddings) instead of torchvision ResNet50 (2048-d). `--method gmm` fits a diagonal Gaussian mixture in PCA space. `--embedding-format both` also writes `embeddings.parquet`, with one column per feature plus the coordinates.

`--max-patches` keeps an even spread of tiles across the slide when a WSI would otherwise produce tens of thousands of patches.

Additional public GBM slides are available from the NCI Genomic Data Commons (TCGA-GBM) and from CPTAC-GBM on TCIA; download those under their data-use terms and pass the local path to `--slide`.

### Offline fallback (`--demo`)

If you cannot download a slide, `python main.py --demo --output-dir outputs/demo` builds a synthetic four-region cartoon and runs the same pipeline. That path does not need OpenSlide on Windows (flat TIFF + in-memory fallback). Prefer the TCGA-GBM example above when you want a real brain-tumor `.svs`.

## Tests

```bash
pip install pytest
pytest
```

The tests cover the entropy math, the tissue mask, and a tiny end-to-end path that uses a randomly initialized ResNet50 so it does not need the ImageNet download. The TCGA-GBM command above (or `python main.py --demo`) is the run that uses pretrained weights.

## How the 20× patch size is chosen

Many diagnostic scans are acquired at 40× (about 0.25 µm per pixel). A 256×256 patch at 20× should cover twice that length on the glass, i.e. 512 level-0 pixels, which are then resized to 256. The slide's `openslide.objective-power` property (or microns-per-pixel, if power is missing) drives that conversion. If a file records neither, level 0 is treated as already being at the requested magnification. The TCGA-06-0743 example records objective power 20 (~0.50 µm/px), so each 256×256 patch is read directly from level 0.

Patches are stored at 256×256. The encoder resizes them to 224×224 because that is the ImageNet / DINOv2 training size (and 224 is divisible by DINOv2's patch size of 14).

## Limitations

- The tissue mask is a classical color threshold. Pen marks, folds, bubbles, and very pale stain can be kept or dropped incorrectly. Look at `tissue_overlay.png` before trusting a score.
- `k` is a hyperparameter. Changing `k` changes the entropy. Compare slides at a fixed `k`. The normalized score makes different values of `k` less misleading, but it does not remove the choice.
- ImageNet ResNet50 is a baseline encoder, not a pathology foundation model. Niches follow whatever visual cues that network uses (color and texture). They are not automatically the niches a neuropathologist would draw.
- Global entropy ignores geography. Two slides can share a score while one is split into large regions and the other is salt-and-pepper. Use `spatial_mixing` and `cluster_heatmap.png` together with the score.
- Embeddings are reduced with PCA (32 dimensions by default) before clustering so K-Means is not dominated by near-duplicate ResNet channels, and so a Gaussian mixture has a covariance it can estimate.
