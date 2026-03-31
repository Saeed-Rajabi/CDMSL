

# CDMSL: Conditional Diffusion Model for Sea-Level Reconstruction

This repository contains the code, and trained models for:

**"CDMSL: High-resolution sea-level reconstruction using conditional diffusion models and SWOT satellite altimetry"**

---

## Overview

Sea level varies at small spatial scales due to ocean dynamics such as eddies, fronts, and coastal processes. While SWOT showed benefits comapred to other observational tools and doscovered new insights, it often lack the spatial and temporal resolution needed to resolve these features.

This project introduces **CDMSL (Conditional Diffusion Model for Sea Level)**, a machine learning framework that reconstructs high-resolution sea-level fields from coarse-resolution sea level data.

---

## Key Features

- Diffusion-based super-resolution of sea levels
- Trained on SWOT satellite observations (∼2 km resolution)
- Reconstructs mesoscale–submesoscale features (~20 km)
- Supports uncertainty quantification via stochastic sampling
- Applicable to regional and global ocean datasets

---

## Repository Structure
cdmsl/
├── main.py
├── train.py
├── inference.py
├── config.py
├── models/
├── diffusion/
├── data/
├── utils/
└── outputs/


---

## Data access

Zenodo archive (data + code):

📌 Original datasets:
- SWOT: https://swot.jpl.nasa.gov/
- ERA5: https://cds.climate.copernicus.eu/

---

## Installation, Usage, Licence, Citation 

```bash
git clone https://github.com/YOUR_USERNAME/cdmsl.git
cd cdmsl

pip install -r requirements.txt

Usage
Train model:
python main.py
Inference

Handled automatically after training, or via loading trained_model and then:

python infer.py

License
Code: MIT License
Data: CC-BY 4.0

## Citation

If you use this work (model/codes), please cite:

zenedo...
