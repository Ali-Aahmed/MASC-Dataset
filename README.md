# MASC Dataset

## 📌 Overview
MASC (Mobile Application Screen Classification) is a dataset designed for automated testing and classification of mobile application interfaces. The dataset contains structured UI representations that can be used for various machine learning tasks such as screen type classification, UI understanding, and automated test case generation.

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
│   ├── feature_extraction.py    # ملف واحد لاستخراج الميزا
│   ├── README.md                # توثيق داخلي للكود
├── data/
│   ├── raw/                      # Original, unprocessed UI data
│   ├── processed/                 # Cleaned and structured dataset
├── README.md                     # Project documentation
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

## 📜 License
This dataset and source code are licensed under the [MIT License](LICENSE).

## 📧 Contact
For questions or collaborations, contact:
**Ali Ahmed** – ali.ahmed.@mu.edu.eg

