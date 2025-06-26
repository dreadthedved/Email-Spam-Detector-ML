# spam_classifier_multi_model.py

# 1. Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# Create a directory to save graphs
os.makedirs("graphs", exist_ok=True)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Plotting setup
sns.set(style="whitegrid")


# 2. Load Dataset
df = pd.read_csv("C:/Users/deves/drone_rl/Email_Spam/spam.csv", encoding='latin-1')[['label', 'text']]
df.columns = ['label', 'message']

# 3. Text Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_message'] = df['message'].apply(clean_text)


# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned_message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)





# 5. Enhanced Training Loop with ROC/AUC
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear)": SVC(kernel='linear',probability=True),  # No probability by default
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = []
roc_data = {}

for name, model in models.items():
    
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Use decision_function or predict_proba
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        y_score = model.decision_function(X_test)

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📌 {name}")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("AUC Score:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "AUC": roc_auc
    })
    
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"graphs/{name}_confusion_matrix.png")
    plt.close()

# 6. Plot All ROC Curves

plt.figure(figsize=(10, 7))

for name, (fpr, tpr, auc_score) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📉 ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/roc_curve_comparison.png")
plt.close()


# 6. Plot Accuracy & F1 Comparison

results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)

# Accuracy
# Accuracy
plt.figure(figsize=(10, 4))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='Set2')
plt.title("Model Accuracy Comparison")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/model_accuracy_comparison.png")
plt.close()

# F1 Score
plt.figure(figsize=(10, 4))
sns.barplot(x='Model', y='F1 Score', data=results_df, palette='Set3')
plt.title("Model F1 Score Comparison")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/model_f1_comparison.png")
plt.close()

# AUC Score
plt.figure(figsize=(10, 4))
sns.barplot(x='Model', y='AUC', data=results_df, palette='coolwarm')
plt.title("Model AUC Score Comparison")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/model_auc_comparison.png")
plt.close()



