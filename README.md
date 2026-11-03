# Bertillot et al. 2026

This repository contains analysis notebooks and helper utilities associated with the CRC tissue architecture, nuclear morphology, and GeoMx spatial transcriptomics analyses from Bertillot et al.

The code is organized as independent analysis modules. Each module can be run separately with the relevant inputs described in the corresponding notebook.

## Repository Contents

```text
Architecture Analysis/
  tumor-stroma_mask_prediction.ipynb
  architecture_training_pipeline.ipynb
  architecture_label_prediction.ipynb
  morphological_feature_extraction.ipynb
  neighborhood_feature_extraction.ipynb
  UNet_utils.py
  architecture_utils.py
  morphology_utils.py
  neighborhood_utils.py

Nuclear Shapes/
  compute_nuclear_shapes.ipynb
  nuclear_shape_utils.py

GeoMx Analysis/
  geoMx_analysis_notebook.ipynb
  geoMx_pseudotime_plots.ipynb
  geomxlib.py

models/
  tumor_stroma_unet.h5
  architecture_unet.h5.part01 ... part04
  join_architecture_unet.py
```

## Analysis Modules

The notebooks are independent modules rather than a single ordered workflow. Run only the modules needed for the analysis being reproduced.

- **Tumor/stroma segmentation**: `Architecture Analysis/tumor-stroma_mask_prediction.ipynb`
  Generate tumor/stroma masks from pan-epithelial images using a pretrained U-Net.
- **Architecture classifier training**: `Architecture Analysis/architecture_training_pipeline.ipynb`
  Train the architecture classifier from U-Net embeddings and labeled tiles.
- **Architecture label prediction**: `Architecture Analysis/architecture_label_prediction.ipynb`
  Apply saved classifier artifacts to embedding JSON files.
- **Architecture morphology**: `Architecture Analysis/morphological_feature_extraction.ipynb`
  Measure manuscript-aligned mask morphology features: branch length, roughness, lumen area, and tumor/stroma summaries.
- **Neighborhood features**: `Architecture Analysis/neighborhood_feature_extraction.ipynb`
  Measure local cell-neighborhood features from centroid/orientation tables.
- **Nuclear shapes**: `Nuclear Shapes/compute_nuclear_shapes.ipynb`
  Measure single-nucleus morphology from labeled nuclear masks.
- **GeoMx analysis**: `GeoMx Analysis/geoMx_analysis_notebook.ipynb`
  Reproduce GeoMx clustering, DGE, and GSEA analyses.
- **GeoMx pseudotime plots**: `GeoMx Analysis/geoMx_pseudotime_plots.ipynb`
  Reproduce pseudotime and spatial projection figures from GeoMx-derived inputs.

## Pretrained Models

The two pretrained U-Nets used in the paper are included in `models/`. Both are ZeroCostDL4Mic 2D multilabel U-Nets (https://github.com/HenriquesLab/ZeroCostDL4Mic) saved with Keras 2.15, taking single-channel `512 x 512` pan-cadherin images as input.

| Model | File | Output | Used by |
|---|---|---|---|
| Tumour–stroma U-Net (trained on 30 TMA cores) | `models/tumor_stroma_unet.h5` | 3 classes: `0` background, `1` stroma, `2` tumour | `tumor-stroma_mask_prediction.ipynb` |
| Architecture U-Net (trained on 196 archetype TMA cores) | `models/architecture_unet.h5` | 5 classes: glandular, fragmented, solid, stroma, edge/background; bottleneck `conv2d_23` gives the 1024-dim tile embeddings | `architecture_training_pipeline.ipynb` |

GitHub limits files to 100 MB, so the architecture U-Net (373 MB) is stored in four parts. Reassemble it once before running the notebooks:

```bash
python models/join_architecture_unet.py
```

The script writes `models/architecture_unet.h5` and checks it against the SHA-256 of the original file:

```text
4bc19bac1439f984c56f6f28f2e5ae3eb0e00b3790e1c525efe7ddb5eda1bfc5  architecture_unet.h5
2e7c069b80743a695b668db05124ab41456269927dfbccdea372fc939c77bab9  tumor_stroma_unet.h5
```

The architecture U-Net is also available as a single file, with ONNX exports and inference code, at https://doi.org/10.5281/zenodo.20343193. The tile-level architecture classifier is not included: as described in the paper Methods, it needs to be retrained for each dataset with `architecture_training_pipeline.ipynb`.

## Installation

The analyses span GPU TensorFlow, GPU/CPU image morphology, and Scanpy/GeoMx workflows. Separate module-specific environments are recommended.

### Conda

Conda is the recommended option for most users because several workflows include compiled or GPU-aware dependencies.

Create an environment for the module you want to run:

```bash
conda env create -f environment-architecture.yml
conda activate architecture-analysis
```

Available environments:

- `environment-architecture.yml`: U-Net segmentation, architecture embeddings, and architecture classification.
- `environment-geomx.yml`: GeoMx spatial transcriptomics analysis, differential expression, and GSEA.
- `environment-neighborhood.yml`: cell-neighborhood distances, orientation order, and stroma proximity features.
- `environment-nuclear-shapes.yml`: nuclear shape feature extraction from labeled nuclear masks.

### Python venv / pip

Users who prefer `venv` can install the same Python packages manually from the `pip:` section of the relevant `environment-*.yml` file:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install tensorflow==2.17.0 keras==3.6.0 numpy==1.26.4
```

This route is often sufficient for CPU-only utilities and lightweight notebook review. For GPU TensorFlow, CuPy, Scanpy, or PyDESeq2 workflows, Conda or an existing institutional/HPC environment is usually more reliable.

## Usage

Open the notebook for the module of interest and edit the configuration cell at the top of the notebook to point to the local input and output paths. A `config_template.json` file is included as an optional template for organizing paths across notebooks.

Example:

```bash
conda env create -f environment-neighborhood.yml
conda activate neighborhood-analysis
jupyter lab
```

## Notes

- Folder names contain spaces to preserve the submitted analysis structure. Quote paths in shell commands.
- Generated files, the trained architecture classifier, and private data are excluded from version control. The pretrained U-Nets are included (see Pretrained Models).
- If U-Net inference on CPU is unexpectedly slow (minutes per 512 px patch), set the environment variable `TF_ENABLE_ONEDNN_OPTS=0` before starting Jupyter.
- `cupy` is only needed for GPU-accelerated nuclear shape extraction.
- `skan` is used for manuscript-aligned skeleton branch distances; the morphology utility falls back to total skeleton length if unavailable.
- `python-louvain` and `networkx` are needed for the corrected centroid-community fragmentation index.

## Citation

If you use this code or the pretrained models, please cite:

> Bertillot, F., Hervé, S., Outla, Z. *et al.* Heterogeneity co-evolves with architecture to determine colon cancer outcome. Manuscript in revision (2026).
>
> Analysis code and pretrained U-Nets: https://doi.org/10.5281/zenodo.20407001
>
> Architecture U-Net with ONNX exports: https://doi.org/10.5281/zenodo.20343193

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Please see `LICENSE` for details.
