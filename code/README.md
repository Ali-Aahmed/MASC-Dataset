# Code Directory Overview

This directory contains the core scripts for feature extraction and classification.

## 📌 Feature Extraction
- **File:** `feature_extraction.py`
- **Description:** This script extracts features from raw UI data, including screenshots, JSON annotations, and view hierarchies.
- **Key Functions:**
  - Extracts structural and semantic features from UI elements.
  - Preprocesses and normalizes UI attributes for machine learning models.

## 📌 Classification
- **File:** `masc_classification.py`
- **Description:** This script implements the classification pipeline for UI screen types.
- **Key Functions:**
  - Loads extracted features.
  - Trains a model for UI classification.
  - Evaluates model performance.

## 🚀 Usage
Run the following commands to extract features and classify UI screens:
```bash
python code/feature_extraction.py
python code/masc_classification.py

## 🚀 Notes
  - feature_extraction.py
  - masc_classification.py
These files will be made publicly available after publication.
