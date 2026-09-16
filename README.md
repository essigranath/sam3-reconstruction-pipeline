# SAM 3 Video Segmentation for 3D Reconstruction

This repository contains the implementation developed for a Bachelor's thesis project investigating the use of SAM 3-based video segmentation as preprocessing for image-based 3D reconstruction.

The project provides a Python-based research environment for processing video material, generating segmentation masks with SAM 3, creating datasets for comparison, producing reference masks with Adobe Photoshop, and running sparse 3D reconstruction with COLMAP.

Three datasets are created and compared:

- Original unsegmented images
- Images with SAM 3 segmentation masks
- Images with reference masks created in Adobe Photoshop

The application also records processing times, active user time, experiment metadata, and COLMAP reconstruction results.

## Features

- MP4 video processing and frame extraction
- SAM 3 video segmentation using text or point prompts
- Automatic creation of comparable frame datasets
- Photoshop reference mask workflow and timing
- Photoshop mask validation
- Sparse 3D reconstruction with COLMAP
- Experiment metadata and logging
- PLY model export
- Reconstruction statistics including registered images, 3D points, and mean reprojection error

## Requirements

The project was developed and tested on Windows 11.

- Python 3.12
- Miniconda or Anaconda
- NVIDIA CUDA-capable GPU
- PyTorch with CUDA support
- SAM 3
- COLMAP
- Adobe Photoshop for the reference segmentation workflow
- Python dependencies listed in `requirements.txt`

> SAM 3, model checkpoints, COLMAP, and Adobe Photoshop are not included in this repository and must be installed separately.

---

## Setup

### Clone the repository

```bash
git clone github.com/essigranath/sam3-reconstruction-pipeline
cd sam3-reconstruction-pipeline
```

### Create the environment

Create and activate a Conda environment for the project:

```bash
conda create -n sam3 python=3.12
conda activate sam3
```

### SAM 3

Clone the official SAM 3 repository into `third_party/sam3`:

```bash
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
```

The project was developed using the SAM 3.1 `sam3.1_multiplex.pt` checkpoint. Download the checkpoint separately from Hugging Face, create a `checkpoints/` directory in the project root, and place the checkpoint there.

### Python dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The SAM 3 package is installed in editable mode as part of the requirements.

### COLMAP

COLMAP must be installed separately.

The current implementation expects COLMAP at:

```text
C:/Tools/COLMAP/COLMAP.bat
```

If COLMAP is installed elsewhere, update the path in `src/colmap_processor.py`.

## Running the Application

Activate the Conda environment:

```bash
conda activate sam3
```

Run the application from the project root directory:

```bash
python scripts/run_gui.py
```

---

## Workflow

1. Select an MP4 video.
2. Define the capture conditions.
3. Enter a text prompt or select a point for SAM 3.
4. Select the frame subset interval.
5. Run frame extraction, SAM 3 segmentation, and dataset creation.
6. Create the reference masks in Adobe Photoshop and save them in the provided folder.
7. Validate the Photoshop masks.
8. Run COLMAP reconstruction for the Original, SAM 3, and Photoshop datasets.
9. Review the generated experiment metadata and reconstruction results.

Each run creates a separate folder under `experiments/` containing the generated datasets, masks, COLMAP results, metadata, and logs.

---

## Repository Notes

Model checkpoints, generated experiment data, and third-party software are not included in the repository. The corresponding directories are excluded from version control:

```text
checkpoints/
experiments/
third_party/
```

## License

This project is licensed under the MIT License.
