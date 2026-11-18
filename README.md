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

data/geomx/
  gene_sets.json
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
  Measure the per-core mask features of the paper from tumour–stroma masks: branch length, branch types and straightness, skeleton cycles, lumens, contour roughness, and tumour–stroma ratio.
- **Neighborhood features**: `Architecture Analysis/neighborhood_feature_extraction.ipynb`
  Measure local cell-neighborhood features from centroid/orientation tables.
- **Nuclear shapes**: `Nuclear Shapes/compute_nuclear_shapes.ipynb`
  Measure the single-nucleus shape features of the paper (PyRadiomics perimeter and axes, contour area, elliptic Fourier descriptors) from labeled nuclear masks.
- **GeoMx analysis**: `GeoMx Analysis/geoMx_analysis_notebook.ipynb`
  Reproduce GeoMx clustering, DGE, and GSEA analyses.
- **GeoMx pseudotime plots**: `GeoMx Analysis/geoMx_pseudotime_plots.ipynb`
  Reproduce pseudotime and spatial projection figures from GeoMx-derived inputs.

## Pretrained Models

The two pretrained U-Nets used in the paper are included in `models/`. Both are 2D multi-label U-Nets saved with Keras 2.15, taking single-channel `512 x 512` pan-cadherin images as input.

| Model | File | Output | Used by |
|---|---|---|---|
| Tumour–stroma U-Net (trained on 30 TMA cores) | `models/tumor_stroma_unet.h5` | 3 classes: `0` background, `1` stroma, `2` tumour | `tumor-stroma_mask_prediction.ipynb` |
| Architecture U-Net (trained on 196 archetype TMA cores) | `models/architecture_unet.h5` | 5 classes: `0` edge/background, `1` stroma, `2` glandular, `3` solid, `4` fragmented; bottleneck `conv2d_23` gives the 1024-dim tile embeddings | `architecture_training_pipeline.ipynb` |

GitHub limits files to 100 MB, so the architecture U-Net (373 MB) is stored in four parts. Reassemble it once before running the notebooks:

```bash
python models/join_architecture_unet.py
```

The script writes `models/architecture_unet.h5` and checks it against the SHA-256 of the original file:

```text
4bc19bac1439f984c56f6f28f2e5ae3eb0e00b3790e1c525efe7ddb5eda1bfc5  architecture_unet.h5
2e7c069b80743a695b668db05124ab41456269927dfbccdea372fc939c77bab9  tumor_stroma_unet.h5
```

The architecture U-Net is also available as a single file, with ONNX exports and inference code, at https://doi.org/10.5281/zenodo.20343193. The tile-level architecture classifier is not included: as described in the paper Methods, it needs to be retrained for each dataset with `architecture_training_pipeline.ipynb`, which saves it to `models/architecture_classifier/`.

## Gene Sets

The gene sets used in the paper are included in `data/geomx/gene_sets.json`. The sources of each panel are listed in the manuscript Methods (gene set enrichment analysis and consensus gene sets).

| Group | Gene sets | Used for |
|---|---|---|
| `STEM_signatures` | Crypt base, LGR5 ISC, Fetal (human), RSC, revCSC, EpiHR, coreHRC | GeoMx GSEA (`geoMx_analysis_notebook.ipynb`) |
| `EMT_signatures` | Wound response, Mesenchymal, Matrix remodeling, TGFB activation, EMT activation, EMT Hallmark, pEMT, EMT-II (MP13) | GeoMx GSEA |
| `CRC_gene_sets` | Consensus lists Core-EMT (`Core EMT`), Core-CBC (`STEM-CBC`) and Core-RegSC (`STEM-RegCS`), plus β-catenin (`CTNNB1_TARGETS`) and YAP (`YAP targets`) target genes | Single-cell and organoid scoring |

"Fetal (human)" contains the robust genes of the human fetal signature.

## Input Data

Patient images, GeoMx data and derived tables are not included (see the paper's Data availability statement). The notebooks expect them under `data/`, following the layout in `config_template.json`. For the GeoMx notebooks:

- `data/geomx/geomx_crc_dataset.h5ad` with `obs` columns `patient_id` (`P1`–`P8`), `segment` (`Segment 1` = tumour) and `architectures` (`solid`, `glandular dense`, `glandular`, `glandular infiltrated`, `mod. fragmented`, `fragmented`)
- `data/geomx/raw_dges_architectures.csv` (differential expression results used for GSEA)
- `data/geomx/pseudotime/P<n>_pseudotime_data.csv` for the three pseudotime patients (`P5`, `P6`, `P8`)

For the architecture notebooks:

- `data/architecture/images/`: single-channel pan-cadherin TIFFs of the TMA cores for the tumour–stroma notebook. The whole-core example of the training notebook (`IMAGE_NAME`) also accepts a two-channel TIFF, in which case channel 1 (pan-cadherin) is used.
- `data/architecture/tiles/`: `448 x 448` pan-cadherin tiles named `<core>_tile_x<x>_y<y>.png`
- `data/architecture/training_labels.csv` with columns `image_name` and `label` (`Glandular`, `Fragmented`, `Solid`, `Stroma`, `Edge`; `Odd` tiles are merged into `Edge`)
- `data/architecture/embeddings/<core>_embedding.json`: tile embeddings (`x`, `y`, 1,024-value `embedding`) for the label prediction notebook

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
- `environment-geomx.yml`: GeoMx spatial transcriptomics analysis, differential expression, and GSEA. It pins Scanpy 1.10.2 and harmonypy 0.0.10 rather than the Scanpy 1.9.6 and Harmony 0.0.9 cited in the manuscript Methods; the differential expression results reproduce with these versions.
- `environment-neighborhood.yml`: architecture morphology features, cell-neighborhood distances, orientation order, and stroma proximity features.
- `environment-nuclear-shapes.yml`: nuclear shape feature extraction from labeled nuclear masks (Python 3.9, PyRadiomics 3.1.0).

### Python venv / pip

Users who prefer `venv` can install the same Python packages manually from the `pip:` section of the relevant `environment-*.yml` file:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
python -m pip install tensorflow==2.17.0 keras==3.6.0 numpy==1.26.4
```

This route is often sufficient for CPU-only utilities and lightweight notebook review. For GPU TensorFlow, Scanpy, or PyDESeq2 workflows, Conda or an existing institutional/HPC environment is usually more reliable.

## Usage

Open the notebook for the module of interest and edit the configuration cell at the top of the notebook (tagged `parameters`) to point to the local input and output paths. By default the paths are relative to the notebook and follow the layout in `config_template.json`.

Example:

```bash
conda env create -f environment-neighborhood.yml
conda activate neighborhood-analysis
jupyter lab
```

The notebooks can also be run headless with [papermill](https://papermill.readthedocs.io), passing the paths as parameters:

```bash
python -m pip install papermill
cd "Architecture Analysis"
papermill neighborhood_feature_extraction.ipynb out.ipynb -p CELL_FEATURE_TABLE cells.csv -p OUT_DIR ../outputs/neighborhood
```

## Notes

- Folder names contain spaces to preserve the submitted analysis structure. Quote paths in shell commands.
- The GeoMx differential expression in the paper compares each architecture against all other tumour regions without a patient term. To adjust for patient, set `DGE_COVARIATES = ['patient_id']` in the notebook, or run it with `papermill geoMx_analysis_notebook.ipynb out.ipynb -y "DGE_COVARIATES: [patient_id]"`. Architecture and patient are strongly confounded in this dataset (for example, fragmented regions come mostly from two patients), so this changes several architecture-level results.
- Generated files, the trained architecture classifier, and private data are excluded from version control. The pretrained U-Nets and the gene sets are included (see Pretrained Models and Gene Sets).
- If U-Net inference on CPU is unexpectedly slow (minutes per 512 px patch), set the environment variable `TF_ENABLE_ONEDNN_OPTS=0` before starting Jupyter.
- `skan` is pinned to 0.11.1 for the skeleton branch measurements; later releases rename the summary columns that the morphology features read.
- The tumour and stroma label values are parameters of the morphology notebook: the masks written by `tumor-stroma_mask_prediction.ipynb` use `1` for stroma and `2` for tumour, while the masks behind the paper tables used `1` for tumour and `2` for stroma.
- `python-louvain` and `networkx` are needed for the corrected centroid-community fragmentation index.

## Citation

If you use this code or the pretrained models, please cite:

> Bertillot, F., Hervé, S., Outla, Z. *et al.* Heterogeneity co-evolves with architecture to determine colon cancer outcome. Manuscript in revision (2026).
>
> Analysis code, pretrained U-Nets and gene sets: https://doi.org/10.5281/zenodo.20407001
>
> Architecture U-Net with ONNX exports: https://doi.org/10.5281/zenodo.20343193

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Please see `LICENSE` for details.
