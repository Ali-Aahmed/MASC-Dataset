# MASC Dataset

## 📌 Overview
**MASC** (Mobile Application Screen Classification) is a **manually curated dataset** containing **7,065 mobile UI screens** classified into **10 distinct categories**. Designed for UI/UX research and ML applications, it enables:
- Accurate screen type classification 📱
- Automated UI testing 🤖
- Design pattern analysis 🎨

## 🌟 Key Features
- **Multi-modal Data**: Screenshots + JSON hierarchies + Semantic annotations
- **High Quality**: 3-step manual validation process ✅
- **ML-Ready**: Pre-extracted feature vectors for 11 UI characteristics

## Design topics overview

| Topic name  | Num. UIs | Description                        |
| :---        |     ---: | :---                               |
| Chat        |      329 | Chat functionality                 |
| List        |      960 | Elements organized in a column     |
| Login       |      889 | Input fields for logging           |
| Maps        |      500 | Geographic display                 |
| Menu        |      557 | Items list in an overlay or aside  |
| Profile     |      526 | Info on a user profile or product  |
| Search      |      725 | Search engine functionality        |
| Settings    |      629 | Controls to change app settings    |
| Welcome     |     1084 | First-run experience               |
| Home        |      163 | home screen                        |

| Total UI=   |     7065 | Onboarding screen                  |

## 📂MASC Dataset
- **Full Dataset**: [Download from kaggle]((https://www.kaggle.com/api/v1/datasets/download/alihmed/masc-dataset))
- **Samples**:  
  [📸 Raw Screenshot](https://github.com/Ali-Aahmed/MASC-Dataset/blob/main/data/raw_samples/315-screenshot.jpg) | 
  [📝 Raw JSON](https://github.com/Ali-Aahmed/MASC-Dataset/blob/main/data/raw_samples/315-screenshot.json) |
  [📊 semantic_JSON](https://github.com/Ali-Aahmed/MASC-Dataset/blob/main/data/raw_samples/315-semantic.json)
  
## 📂 Dataset Structure
The dataset is organized into multiple components, each representing a different aspect of the UI:

- **Screenshot Images:** High-resolution images (JPG, 540x960 px) capturing the visual design of mobile UIs.
- **UI Semantic Annotations (JSON):** A JSON file describing all UI components, including buttons, text fields, and icons.
- **View Hierarchies (JSON):** A DOM-like structure representing parent-child relationships between UI components.
- **MASC_Features.csv (CSV):** File containing extracted features for each UI.
- **Labels.csv (CSV):** File containing  (Screen Id,class) each UI.

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
├── README.md                     # Project documentation
├── LICENSE                        # Usage license
```

## 📥 Accessing the Data
### 1. Public Datasets Used
| Dataset | Description | Link |
|---------|-------------|------|
| **Rico** | 72k Android UI screens | [Download](https://interactionmining.org/rico) |
| **Enrico** | 1,460 curated screens | [GitHub](https://github.com/luileito/enrico) |
| **Screen2Words** | 112k UI descriptions | [Download](https://github.com/google-research/google-research/tree/master/screen2words) |

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
## Citation
If you use this dataset or code in your research, please cite it as follows:

**Ahmed, A. (2025).** "MASC Dataset: A Novel Resource for Classifying Mobile Application Screens using Machine Learning." *Currently under review at The Visual Computer.*  
Available at: [GitHub Repository](https://github.com/Ali-Aahmed/MASC-Dataset)  
DOI: [10.5281/zenodo.14783065](https://doi.org/10.5281/zenodo.14783065)


## 📜 License
This dataset and source code are licensed under the [MIT License](LICENSE).

## 📧 Contact
For questions or collaborations, contact:
**Ali Ahmed** – ali.ahmed.@mu.edu.eg
