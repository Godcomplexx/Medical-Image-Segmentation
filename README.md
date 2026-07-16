# Medical Image Segmentation Service

## Overview

AI-powered medical imaging platform for automated brain MRI analysis and volumetric assessment.
Designed as a production-ready microservice system for clinical and research use.

## Problem

- Manual analysis of brain MRI is time-consuming and subjective
- Neurologists need fast, accurate volumetric measurements of brain structures
- No unified tool that combines DICOM processing, segmentation, and result visualization

## Solution

End-to-end AI-driven medical imaging pipeline with asynchronous processing and scalable microservice architecture:

**DICOM Upload → Preprocessing → YOLO Detection → FastSurfer Segmentation → Volume Calculation → Interactive Visualization**

## Key Features

- Fully asynchronous processing (RabbitMQ + Celery)
- Scalable microservice architecture (11 containers)
- Automated DICOM → segmentation → volumetric analysis pipeline
- GPU-accelerated brain segmentation (FastSurfer)
- Interactive web UI for visualization and patient management

## Architecture

![Project Architecture](images/structure_prj.png)

```
User ──► Nginx (reverse proxy)
              ├──► React Frontend (Vite + MUI)
              └──► FastAPI Backend
                      ├──► PostgreSQL (patient metadata)
                      └──► RabbitMQ ──► Celery Workers
                                            ├──► Math App (DICOM conversion, YOLO)
                                            └──► FastSurfer (GPU segmentation)
```

### Processing Pipeline

![Processing Pipeline](mq/docs/sxem.png)

### Database Schema

![Database Schema](images/structure_bd.png)

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, TypeScript, Vite, Material UI |
| **Backend** | Python, FastAPI, SQLAlchemy, Alembic |
| **ML Models** | YOLO (detection), FastSurfer (segmentation) |
| **Async Processing** | Celery, RabbitMQ, Flower (monitoring) |
| **Database** | PostgreSQL, asyncpg |
| **Infrastructure** | Docker Compose, Nginx, pgAdmin |
| **GPU** | NVIDIA CUDA (FastSurfer container) |

## Results

### Lesion detection and anatomical classification (two-stage cascade)

The core research contribution is a **two-stage cascade** for MS demyelination lesions on FLAIR MRI:
a binary lesion **detector** (2.5D U-Net with a ResNet34 encoder, Dice + Focal-Tversky loss) followed
by an **anatomy-aware classifier** that assigns each detected lesion a McDonald class — I (infratentorial),
P (periventricular), S (subcortical), U (juxtacortical) — from brain-structure context (SynthSeg masks),
instead of the lesion's local appearance alone.

Evaluated on independent, study-level held-out patients:

| Stage | Metric | Value |
|-------|--------|-------|
| Classification (given correct localization) | macro-F1 | **0.84** (I 0.92 / P 0.79 / S 0.86 / U 0.80) |
| Detector | AUC-ROC / AP (pixel-level) | **0.85 / 0.73** |
| Detector | pixel Dice / lesion recall | 0.72 / 0.67 |
| End-to-end cascade | macro-recall | 0.45–0.59 (U up to **0.78** on lesions ≥100 px) |

**Key finding.** Ablation (identical full recipe, one factor changed at a time) shows that SynthSeg
structure features add **+0.18 macro-F1** to the classifier and the Focal-Tversky loss term adds
**+0.41 precision** (plus training stability) over plain Dice; the 2.5D input has a negligible effect
vs 2D. A fully auto-configured 3D **nnU-Net** on the same unmodified annotation reaches macro-recall
0.37 and does **not** beat the cascade — the quality ceiling is annotation completeness, not the
network architecture.

Full methodology, all figures (qualitative examples, feature importance, ROC/PR) and metrics with
95% confidence intervals are in the manuscript under [`docs/`](docs/)
(`МРТ_РС_ИИ_детекция_классификация_очагов.docx`).

<details>
<summary>Early single-stage YOLO baseline (superseded by the cascade)</summary>

Initial single-stage YOLO segmentation (4 classes I/P/S/U, imgsz 640) — per-class precision before/after
fine-tuning. It plateaued because a lesion's anatomical class is defined by its position relative to brain
structures, not by its local texture, which motivated the two-stage anatomy-aware design above.

| Class | Precision (v1) | Precision (v2) |
|-------|:--------------:|:--------------:|
| I | 0.64 | 0.70 |
| P | 0.79 | 0.84 |
| S | 0.71 | 0.78 |
| U | 0.72 | 0.76 |

<p align="center">
  <img src="images/photo_2025-11-17_08-22-10.jpg" width="45%" alt="Confusion Matrix v1"/>
  <img src="images/photo_2025-11-27_08-06-50.jpg" width="45%" alt="Confusion Matrix v2"/>
</p>
</details>

### Clinical annotation tool (model-assisted labeling)

A one-click Windows tool lets a radiologist pre-annotate lesions and verify them in **3D Slicer**,
closing the loop back into training data:

1. **`Разметка_очагов_GUI.bat`** — pick a folder of DICOM studies (or FLAIR `.nii/.nii.gz`), tick the
   studies to process, and run. The tool detects lesions (nnU-Net ∪ 2.5D detector) and pre-assigns
   I/P/S/U (anatomy classifier), writing a Slicer-ready segmentation per study.
2. The doctor reviews and corrects in **3D Slicer** (colored, named I/P/S/U segments) and adds any
   missed lesions.
3. **`Принять_правки.bat`** — converts the corrected segmentations back into training-format labels.

End-user guide: [`docs/Приложение_для_врача_описание.docx`](docs/). Code: [`review_tool/`](review_tool/).

## Impact

- Enables automated and reproducible analysis of brain MRI
- Reduces time required for lesion analysis from tens of minutes to seconds
- Supports clinical decision-making with quantitative volumetric data
- Designed for integration into clinical workflows (PACS/RIS systems)

## My Contribution

- Designed microservice architecture (11 containers orchestrated via Docker Compose)
- Implemented FastAPI backend with async database operations
- Built asynchronous ML pipeline: RabbitMQ → Celery workers → YOLO + FastSurfer
- Fine-tuned YOLO model for brain MRI segmentation, improving precision across all classes
- Developed React frontend for patient management and result visualization
- Configured Nginx reverse proxy and container networking
- Integrated GPU-accelerated FastSurfer for brain segmentation

## Quick Start

```bash
# Model weights (*.pt) are stored via Git LFS — install it BEFORE cloning
git lfs install

# Clone and configure
git clone <repository-url>
cd Medical-Image-Segmentation
git lfs pull                 # fetch best.pt model weights
cp .env.example .env         # REQUIRED: set POSTGRES_PASSWORD, JWT_SECRET_KEY,
                             # INTERNAL_API_TOKEN, PGADMIN_PASSWORD,
                             # RABBITMQ_USER and RABBITMQ_PASS

# Build and run
docker-compose up --build -d

# Check status
docker-compose ps
```

> **Model weights & Git LFS.** `best.pt` is identical for `finetune/` and `math_app/`;
> both are tracked by Git LFS (see [.gitattributes](.gitattributes)) and, being byte-identical,
> are stored as a **single LFS object** (one OID), eliminating duplicate storage.
> To migrate the weights already present in earlier history into LFS, run once and force-push:
> `git lfs migrate import --include="*.pt" --everything`.

Open `http://localhost` in your browser. See [USAGE.md](USAGE.md) for detailed usage instructions.

## Project Structure

```
├── api/            # FastAPI backend service
├── front/          # React frontend (Vite)
├── math_app/       # ML computation service (YOLO, DICOM processing)
├── fastsurfer/     # Brain segmentation model (GPU)
├── mq/             # RabbitMQ config + Celery workers
├── nginx/          # Reverse proxy configuration
├── config/         # Shared Python config
├── finetune/       # Research: two-stage lesion pipeline (detector + anatomy classifier), training code
├── review_tool/    # Doctor annotation tool (GUI, DICOM→pre-labels→3D Slicer→training)
├── docs/           # Manuscript, technical reports, end-user guide, figures
├── images/         # Architecture diagrams and results
├── Разметка_очагов_GUI.bat      # Launch the annotation tool (GUI)
├── Принять_правки.bat           # Import doctor-verified labels back into training
└── docker-compose.yml
```

## Demo data & licenses

### Demo study `689`

A pre-processed demonstration study is committed at
[`math_app/fileStorage/processed_data/689`](math_app/fileStorage/processed_data/689) for output
format validation and download/viewer development. It is not raw DICOM input and therefore cannot
exercise the DICOM-to-segmentation pipeline by itself. A full processing run requires a permitted
DICOM study uploaded through the application. The demo contains four NIfTI volumes:

| File | Meaning |
|------|---------|
| `T1.nii` | T1-weighted volume |
| `T2.nii` | T2-weighted volume |
| `Predict/segmentation.nii` | YOLO/segmentation result |
| `ParcelationMask/parcelation.nii` | FastSurfer parcellation mask |

**Privacy note:** these are processed NIfTI volumes, so the original DICOM demographic fields are
not present. That fact alone does not prove complete anonymization of medical imagery; the project
maintainer is responsible for confirming that the example is authorized for distribution and,
where required, defaced. The directory name `689` is treated as an opaque demo code. The set is
kept in version control intentionally and is **excluded from automated test-data cleanup** (tests
use `tmp_path`/their own fixtures and never touch `fileStorage/processed_data/689`). Automated
smoke tests verify the NIfTI signatures and ensure the `descrip` and `aux_file` header fields are
empty; this is an integrity/metadata check, not a clinical de-identification certification.

### FreeSurfer / FastSurfer license

The FastSurfer license is committed at
[`math_app/fileStorage/fast_surfer_lic/license.txt`](math_app/fileStorage/fast_surfer_lic/license.txt).
At its documented path it is mounted **read-only** into the FastSurfer container
(`/fileStorage/fast_surfer_lic:ro` in [docker-compose.yml](docker-compose.yml)); the rest of
`fileStorage` stays writable so FastSurfer can write its outputs.

### Model weights

`best.pt` is tracked via Git LFS and de-duplicated to a single object — see the Quick Start above
and [docs/DEDUP_NOTES.md](docs/DEDUP_NOTES.md).
