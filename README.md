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
 
## Citation
If you use this code, please cite:
```
Isa, A., Boukari, S., & Aliyu, M. (2026). FCA-Guided Counterfactual Explanations for Multi-Modal Breast Cancer Diagnosis: A Framework Achieving Perfect Validity with Emergent Sparsity. Intelligent Oncology.

```
 
## Licence
MIT. See LICENSE file.
