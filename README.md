# 🩺 Breast Cancer Diagnostic System
### Machine Learning · Python · Streamlit

A machine learning-powered web application that classifies breast tumor measurements as **Benign** or **Malignant** using supervised learning algorithms. The project demonstrates model training, performance evaluation, and interactive data exploration through a Streamlit dashboard.

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/Status-Educational%20Project-success" alt="Project status" />

</p>

**Repository:** [Breast-Cancer-Diagnostic](https://github.com/divyasingh97/Breast-Cancer-Diagnostic)

---

## 📌 Overview

Breast cancer classification is a binary classification problem in which tumor measurements are used to predict one of two categories: benign or malignant.

This project uses the **Breast Cancer Wisconsin dataset** available through Scikit-learn to build and compare machine learning models. A Streamlit interface makes the trained model and its evaluation results accessible through a web-based dashboard.

The project covers the complete basic machine learning workflow:

- Dataset loading and exploration
- Data preprocessing
- Model training and comparison
- Performance evaluation
- Interactive application development

## ✨ Key Features

| Feature | Description |
|---|---|
| 🖥️ Interactive Dashboard | Explore the project through a Streamlit interface |
| 🔬 Tumor Classification | Predict benign or malignant categories from input measurements |
| 🤖 Multiple ML Models | Compare Logistic Regression, Decision Tree, and Random Forest |
| 📊 Model Analytics | Review classification metrics and model performance |
| 🗂️ Dataset Explorer | Explore dataset features and observations |
| 💾 Saved Models | Reuse trained models without retraining every time |
| 📈 Evaluation Metrics | Examine accuracy, precision, recall, F1-score, and ROC-AUC |

## 🛠️ Tech Stack

- **Language:** Python
- **Web Application:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Version Control:** Git and GitHub

## 🧠 Machine Learning Models

The project trains and compares three supervised classification algorithms.

| Model | Purpose |
|---|---|
| Logistic Regression | A baseline classification model |
| Decision Tree | Learns decision rules from feature values |
| Random Forest | Combines multiple decision trees for prediction |

Models are evaluated using multiple metrics rather than relying on accuracy alone. The training script selects a model using malignant-class recall and F1-score, as implemented in the project.

### 📊 Evaluation Metrics

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Of the cases predicted as malignant, the proportion that are actually malignant.
- **Recall:** Of the actual malignant cases, the proportion correctly identified by the model.
- **F1-score:** A combined measure of precision and recall.
- **ROC-AUC:** Measures how well the model separates the two classes across classification thresholds.

> **Important:** These are machine learning evaluation metrics, not evidence of clinical safety or diagnostic reliability.

## 📂 Project Structure

```text
Breast-Cancer-Diagnostic/
│
├── app.py                         # Streamlit web application
├── train_model.py                 # Model training and evaluation
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files excluded from Git
│
└── models/
    ├── all_models.joblib          # Saved trained models
    ├── best_model.joblib          # Selected model
    ├── feature_names.joblib       # Model input feature names
    ├── model_comparison.csv       # Model comparison results
    └── test_metrics.csv           # Test-set evaluation metrics
```

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/divyasingh97/Breast-Cancer-Diagnostic.git
cd Breast-Cancer-Diagnostic
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows — Git Bash:**

```bash
source .venv/Scripts/activate
```

**Windows — Command Prompt:**

```cmd
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local address shown in the terminal. By default, it is:

`http://localhost:8501`

### 5. Retrain the models (optional)

To train the models again and regenerate the saved artifacts, run:

```bash
python train_model.py
```

The training script saves the generated model files and evaluation results in the `models/` directory.

## 🔍 How It Works

1. **Load the dataset:** Obtain the breast cancer dataset provided by Scikit-learn.
2. **Prepare the data:** Separate the input features from the target labels and prepare the training and test sets.
3. **Train models:** Fit Logistic Regression, Decision Tree, and Random Forest classifiers.
4. **Evaluate performance:** Compare models using cross-validation and held-out test metrics.
5. **Select a model:** Choose the model according to the selection criteria implemented in the training script.
6. **Explore predictions:** Use the Streamlit dashboard to interact with the model and review evaluation results.

## 🎯 Learning Outcomes

Through this project, I explored:

- Supervised machine learning and binary classification
- Data preprocessing and feature handling
- Cross-validation and train-test evaluation
- Model comparison using multiple metrics
- Saving and loading trained models with Joblib
- Building interactive machine learning applications with Streamlit
- Managing project code with Git and GitHub

## 🔮 Future Improvements

- Deploy the application to a public hosting platform
- Add automated tests for model predictions and application functionality
- Improve input validation and error handling
- Add more detailed visualizations and model interpretability
- Evaluate performance on additional independent datasets

## 👩‍💻 Author

**Divya Jadon**  
BCA — Artificial Intelligence and Machine Learning  
Galgotias University

- **GitHub:** [@divyasingh97](https://github.com/divyasingh97)
- **Project Repository:** [Breast Cancer Diagnostic System](https://github.com/divyasingh97/Breast-Cancer-Diagnostic)

## ⚠️ Medical Disclaimer

This project is intended **for educational and demonstration purposes only**. It is not a medical device and has not been established as clinically validated. Predictions may be incorrect and must not be used for diagnosis, screening, treatment, or other medical decisions. Consult a qualified healthcare professional for medical advice.

---

<p align="center">
  Built with Python, Scikit-learn, and Streamlit.
</p>
