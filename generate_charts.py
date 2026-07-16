import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

# =========================
# LOAD MODELS
# =========================
with open("all_models.pkl", "rb") as f:
    models = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Articles.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

TEXT = "Article"
LABEL = df.columns[0]

X = df[TEXT].astype(str)
y = df[LABEL].astype(str)

# =========================
# SPLIT SAME AS TRAIN
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_tfidf = tfidf.transform(X_test)

# =========================
# STORAGE
# =========================
accuracy = {}
f1_scores = {}
precision = {}
recall = {}

best_model_name = ""
best_acc = 0
best_model = None

# =========================
# EVALUATE MODELS
# =========================
for name, model in models.items():

    y_pred = model.predict(X_test_tfidf)

    acc = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    accuracy[name] = acc
    f1_scores[name] = f1
    precision[name] = prec
    recall[name] = rec

    print(f"{name} -> Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

# =========================
# 1. ACCURACY CHART
# =========================
plt.figure(figsize=(10,5))
plt.bar(accuracy.keys(), accuracy.values())
plt.title("Accuracy Comparison")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("accuracy_chart.png")
plt.show()

# =========================
# 2. F1 SCORE CHART
# =========================
plt.figure(figsize=(10,5))
plt.bar(f1_scores.keys(), f1_scores.values())
plt.title("F1 Score Comparison")
plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("f1_chart.png")
plt.show()

# =========================
# 3. PRECISION CHART
# =========================
plt.figure(figsize=(10,5))
plt.bar(precision.keys(), precision.values())
plt.title("Precision Comparison")
plt.xticks(rotation=45)
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("precision_chart.png")
plt.show()

# =========================
# 4. RECALL CHART
# =========================
plt.figure(figsize=(10,5))
plt.bar(recall.keys(), recall.values())
plt.title("Recall Comparison")
plt.xticks(rotation=45)
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig("recall_chart.png")
plt.show()

# =========================
# 5. CONFUSION MATRIX (BEST MODEL)
# =========================
y_pred_best = best_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# 6. CATEGORY DISTRIBUTION
# =========================
df[LABEL].value_counts().plot(kind="bar", figsize=(10,5))
plt.title("Category Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("category_distribution.png")
plt.show()

# =========================
# FINAL OUTPUT
# =========================
print("\nBEST MODEL:", best_model_name)
print("BEST ACCURACY:", best_acc)

print("\nALL CHARTS SAVED:")
print("accuracy_chart.png")
print("f1_chart.png")
print("precision_chart.png")
print("recall_chart.png")
print("confusion_matrix.png")
print("category_distribution.png")