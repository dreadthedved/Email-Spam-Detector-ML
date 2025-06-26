# 📧 Email Spam Classifier with Multiple ML Models

This project is a machine learning-based system for classifying email messages as **spam** or **ham (not spam)**. It compares multiple algorithms using performance metrics and visualizations such as confusion matrices, ROC curves, and bar charts for accuracy, F1 score, and AUC.

---

## 🔍 Features

- Preprocessing using **NLTK**: cleaning, stopword removal, stemming
- Feature extraction using **TF-IDF**
- Multiple classifier support:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (Linear)
  - Random Forest
  - k-Nearest Neighbors
- Evaluation with:
  - Accuracy
  - F1 Score
  - AUC (Area Under ROC Curve)
  - Confusion Matrices
  - ROC Curve visualization
- Auto-saving all visualizations to `graphs/` folder

---

## 📁 Dataset

The dataset used is [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from the UCI SMS Spam Collection. It contains two columns:
- `label`: `ham` or `spam`
- `text`: the email/SMS content

---

## 🧰 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier

