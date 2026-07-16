import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

german_stopwords = {
    "der", "die", "das", "und", "ist", "im", "in", "am", "an",
    "ein", "eine", "mit", "von", "zu", "den", "des", "auf",
    "für", "bei", "als", "auch", "nicht", "noch", "es", "er",
    "sie", "wir", "ihr", "ich", "du", "man", "oder", "aber",
    "so", "wenn", "wie", "nach", "aus", "dem", "dass", "hat",
    "haben", "war", "wird", "sind"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in german_stopwords]

    return " ".join(words)

print("=" * 80)
print("GERMAN NEWS ARTICLE CLASSIFICATION")
print("=" * 80)

df = pd.read_csv("Articles.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

TEXT_COLUMN = "Article"
LABEL_COLUMN = df.columns[0]

print("\nTotal Articles:", len(df))

print("\nTotal Categories:", df[LABEL_COLUMN].nunique())

print("\nCategory Wise Count:\n")
category_counts = df[LABEL_COLUMN].value_counts()

for category, count in category_counts.items():
    print(f"{category} : {count}")

df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200)
}

trained_models = {}

report_file = open("models_report.txt", "w", encoding="utf-8")

report_file.write("GERMAN NEWS ARTICLE CLASSIFICATION REPORT\n\n")
report_file.write(f"Total Articles: {len(df)}\n")
report_file.write(f"Total Categories: {df[LABEL_COLUMN].nunique()}\n\n")
report_file.write("Category Wise Count:\n\n")

for category, count in category_counts.items():
    report_file.write(f"{category} : {count}\n")

report_file.write("\n\n")

for model_name, model in models.items():
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    try:
        model.fit(X_train_tfidf, y_train)

        trained_models[model_name] = model

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        report_file.write("=" * 80 + "\n")
        report_file.write(f"{model_name}\n")
        report_file.write("=" * 80 + "\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n")
        report_file.write(f"Precision: {precision:.4f}\n")
        report_file.write(f"Recall: {recall:.4f}\n")
        report_file.write(f"F1 Score: {f1:.4f}\n\n")
        report_file.write(report)
        report_file.write("\n\n")

    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

report_file.close()

with open("all_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("\nSaved Files:")
print("all_models.pkl")
print("tfidf.pkl")
print("models_report.txt")
