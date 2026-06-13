# FCA-Guided-CF-TCGA
# FCA-Guided Counterfactual Explanations for Multi-Modal Breast Cancer Diagnosis: A Framework Achieving Perfect Validity with Emergent Sparsity
> Abdullahi Isa, Souley Boukari, Muhammad Aliyu
> Faculty of Computing, Department of Computer Science, Abubakar Tafawa Balewa University, Bauchi, Nigeria. (ATBU Bauchi | June 2026)

 
## Overview
This repository contains the complete implementation of the FCA-Guided
Counterfactual (FCA-CF) framework for multi-modal breast cancer diagnosis.
The framework achieves **Validity = 1.0000**, **Sparsity = 2.37** (best of all
valid methods), and **Proximity = 0.900** on the TCGA-BRCA dataset.
 
## Quick Start (Synthetic Data Mode)
```bash
git clone https://github.com/isaabdullahi2008/FCA-Guided_CF-TCGA.git
cd FCA-Guided-CF-TCGA
pip install -r requirements.txt
python fca_cf_v5.py
```
Results and figures are saved to `tcga_results_v5/`.
 
## TCGA-BRCA Real Data
See `data/README_DATA.md` for download instructions.
The GDC Data Transfer Tool (free) is required. Dataset size: ~150-200 GB.
 
## Results
| Method | Validity | Sparsity | Proximity |
|--------|----------|----------|-----------|
| FCA-Guided CF (Proposed) | **1.000** | **2.37** | **0.900** |
| DiCE | 0.333 | 4.95 | 0.917 |
| Wachter-style | 0.983 | 61.6 | 0.892 |
| FACE | 1.000 | 55.7 | 0.781 |
| NICE | 1.000 | 11.87 | 0.900 |

## Create the destination folders
mkdir C:\TCGA_BRCA_Analysis
mkdir C:\TCGA_BRCA_Analysis\slides
mkdir C:\TCGA_BRCA_Analysis\clinical
 
# Download slides (SVS images) using your manifest
gdc-client download ^
  --manifest C:\TCGA_BRCA\gdc_manifest.txt ^
  --dir C:\TCGA_BRCA_Analysis\slides ^
  --n-processes 4 ^
  --retry-amount 5
 
# Note: ^ is the Windows line-continuation character
# Replace the path with wherever you saved gdc_manifest.txt

# This command is safe to run again after interruption:
gdc-client download ^
  --manifest C:\TCGA_BRCA\gdc_manifest.txt ^
  --dir C:\TCGA_BRCA_Analysis\slides ^
  --n-processes 4 ^
  --retry-amount 5

gdc-client download ^
  --manifest C:\TCGA_BRCA\gdc_manifest_clinical.txt ^
  --dir C:\TCGA_BRCA_Analysis\clinical ^
  --n-processes 4
# In fca_cf_v5.py, find and update these two lines:
 
# BEFORE (default simulation mode):
TCGA_SLIDE_DIR  = Path("C:/TCGA_BRCA_Analysis/slides")
TCGA_CLIN_DIR   = Path("C:/TCGA_BRCA_Analysis/clinical")
 
# The paths above are already correct if you followed Step 10.
# The code will automatically detect if the SVS files are present.
# If they are found, it uses real data.
# If not found, it falls back to the synthetic simulation automatically.

# TCGA-BRCA Dataset Download Instructions
 
## Why the data is not in this repository
The TCGA-BRCA dataset is 150-200 GB and cannot be stored on GitHub.
It is freely available from the NCI GDC portal.
 
## Step 1: Install the GDC Data Transfer Tool
Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
 
## Step 2: Build your manifest on the GDC Portal
1. Go to: https://portal.gdc.cancer.gov/exploration
2. Filter: Project = TCGA-BRCA, Data Format = SVS (slides) + XML (clinical)
3. Click 'Manifest' to download gdc_manifest.txt
## Step 3: Download
```bash
gdc-client download \
  --manifest gdc_manifest.txt \
  --dir C:/TCGA_BRCA_Analysis/slides \
  --n-processes 4 --retry-amount 5
```
 
## Step 4: Configure fca_cf_v5.py
Update TCGA_SLIDE_DIR and TCGA_CLIN_DIR in fca_cf_v5.py to point
to your download location. The code auto-detects real vs. synthetic mode.
 
## Data citation
TCGA Research Network (2012). Comprehensive molecular portraits of human
breast tumours. Nature, 490, 61-70. https://doi.org/10.1038/nature11412
 
## No account required
TCGA-BRCA image and clinical data is open-access. No NIH login needed.

 
## Citation
If you use this code, please cite:
```
Isa, A., Boukari, S., & Aliyu, M. (2026). FCA-Guided Counterfactual Explanations for Multi-Modal Breast Cancer Diagnosis: A Framework Achieving Perfect Validity with Emergent Sparsity. Intelligent Oncology.

```
 
## Licence
MIT. See LICENSE file.
