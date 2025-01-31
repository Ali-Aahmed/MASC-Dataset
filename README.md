# MASC Dataset

## 📌 Overview
MASC (Mobile Application Screen Classification) is a dataset designed for automated testing and classification of mobile application interfaces. It contains structured UI representations that can be used for various machine learning tasks such as screen type classification, UI understanding, and automated test case generation.

## 📂 Dataset Structure
The dataset is organized into multiple components, each representing a different aspect of the UI:

- **Screenshot Images:** High-resolution images (JPG, 540x960 px) capturing the visual design of mobile UIs.
- **Semantic Wireframe Images:** PNG representations of the UI layout, abstracting stylistic elements to focus on structure.
- **UI Semantic Annotations (JSON):** A JSON file describing all UI components, including buttons, text fields, and icons.
- **View Hierarchies (JSON):** A DOM-like structure representing parent-child relationships between UI components.

## 📁 Repository Structure
```
MASC-Dataset/
├── code/
│   ├── masc_classification.py    # Main script for data preprocessing and classification
│   ├── requirements.txt          # List of dependencies
│   ├── feature_extraction.py     # Script for extracting UI features
│   ├── README.md                 # Documentation
├── data/
│   ├── raw/                      # Original, unprocessed UI data
│   ├── processed/                 # Cleaned and structured dataset
├── samples/                      # Example files from the dataset
│   ├── sample_screenshot.jpg      # Sample UI screenshot
│   ├── sample_wireframe.png       # Sample wireframe image
│   ├── sample_hierarchy.json      # Sample view hierarchy
│   ├── sample_annotations.json    # Sample UI annotations
├── README.md                      # Project documentation
├── LICENSE                        # Usage license
```

## 📥 Installation & Setup
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r code/requirements.txt
```

## 🚀 Usage
To preprocess data and train the classification model, run:
```bash
python code/masc_classification.py
```

## 🛠 Dependencies
The project uses the following Python libraries:
```text
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
xgboost==1.7.6
matplotlib==3.7.1
seaborn==0.12.2
nltk==3.8.1
joblib==1.2.0
```

## 📊 Related Datasets
Here are some additional datasets that can complement MASC:
1. **[Rico Dataset](https://interactionmining.org/rico)** – A large-scale mobile app UI dataset.
2. **[Enrico Dataset](http://userinterfaces.aalto.fi/enrico/)** – A curated subset of Rico with UI topic classification.
3. **[GUIs Dataset](https://github.com/google-research-datasets/guicommon)** – A dataset for UI understanding and automation.

## 🔗 Dataset & Samples
- **Full MASC Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/alihmed/masc-dataset)
- **Sample Files:**
  - [Screenshot Sample](samples/sample_screenshot.jpg)
  - [Wireframe Sample](samples/sample_wireframe.png)
  - [Hierarchy Sample](samples/sample_hierarchy.json)
  - [Annotations Sample](samples/sample_annotations.json)

## 📜 License
This dataset and source code are licensed under the [MIT License](LICENSE).

## 📧 Contact
For questions or collaborations, contact:
**Ali Ahmed** – ali.ahmed.@mu.edu.eg

