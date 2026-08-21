"""
train_model.py
Loads Fake.csv + True.csv, trains and compares Logistic Regression,
Naive Bayes, and Random Forest on TF-IDF features, and saves the
best-performing model + vectorizer to models/.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import clean_text

DATA_DIR = "data"
MODELS_DIR = "models"


def load_data():
    fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
    real = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

    fake["label"] = 0  # 0 = Fake
    real["label"] = 1  # 1 = Real

    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna(""))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    return acc


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Cleaning text (this can take a few minutes on the full dataset)...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    best_name, best_model, best_acc = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        acc = evaluate(name, model, X_test_vec, y_test)
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    print(f"Saved to {MODELS_DIR}/best_model.pkl and {MODELS_DIR}/vectorizer.pkl")


if __name__ == "__main__":
    main()
