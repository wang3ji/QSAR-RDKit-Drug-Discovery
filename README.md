# QSAR Modeling of CHEMBL251 (A2A Adenosine Receptor) Using ChEMBL, RDKit, and SHAP

This project builds a full QSAR pipeline for **CHEMBL251 (A2A adenosine receptor)** using IC50 bioactivity data from ChEMBL.  
The workflow includes data retrieval, cleaning, molecular featurization, model training, scaffold splitting, hyperparameter tuning, and model interpretation using SHAP.

---

## 🔍 Project Overview

This repository contains a Jupyter notebook that demonstrates:

- Downloading IC50 bioactivity data for **CHEMBL251** using the ChEMBL Python API  
- Cleaning and standardizing the dataset  
- Converting IC50 values to **pIC50**  
- Generating molecular fingerprints (Morgan/ECFP)  
- Training regression models:
  - **RandomForestRegressor**
  - **XGBRegressor**
- Evaluating models using **R²**  
- Performing **scaffold split** to test generalization  
- Hyperparameter tuning  
- Model interpretation using **SHAP** to identify important fingerprint bits  
- Mapping SHAP‑important bits back to chemical substructures using RDKit  

---

## 📦 Dependencies

Main libraries used:

- Python 3.x  
- RDKit  
- ChEMBL Web Resource Client  
- scikit‑learn  
- XGBoost  
- pandas / numpy  
- SHAP  
- matplotlib / seaborn  

Install using:

```bash
conda install -c conda-forge rdkit
pip install chembl-webresource-client xgboost shap
