# NLU Assignment - Problem-01 : Sports vs Politics 

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------

# Selecting only sports and politics related categories from 20 newsgroups dataset
categories = [
    'rec.sport.baseball',
    'rec.sport.hockey',
    'talk.politics.mideast',
    'talk.politics.misc'
]

# Loading dataset and removing metadata (headers, signatures, quoted replies)
dataset = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

texts = dataset.data

# Converting 4 topic labels into binary classes: SPORT and POLITICS
labels = []
for t in dataset.target:
    if t in [0, 1]:
        labels.append("SPORT")
    else:
        labels.append("POLITICS")

# Display dataset size and available classes
print("Total documents:", len(texts))
print("Classes:", set(labels))

# -----------------------------
# 2. Train-test split
# -----------------------------

# Splitting dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Feature Extraction (TF-IDF + bigrams)
# -----------------------------

# Converting text into numerical vectors using TF-IDF with unigrams and bigrams
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3
)

# Learn vocabulary from training data and transform both train and test text
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

results_summary = {}
results_text = "SPORT vs POLITICS CLASSIFICATION RESULTS\n\n"

# -----------------------------
# 4. Naive Bayes
# -----------------------------

# Training Naive Bayes classifier (probabilistic baseline model)
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# Predicting labels for unseen test documents
nb_pred = nb.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Results")
print("Accuracy:", nb_acc)
print(classification_report(y_test, nb_pred))

results_summary["Naive Bayes"] = nb_acc
results_text += "Naive Bayes Accuracy: " + str(nb_acc) + "\n"
results_text += classification_report(y_test, nb_pred) + "\n"

# -----------------------------
# 5. Logistic Regression
# -----------------------------

# Training Logistic Regression (linear decision boundary model)
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Results")
print("Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))

results_summary["Logistic Regression"] = lr_acc
results_text += "\nLogistic Regression Accuracy: " + str(lr_acc) + "\n"
results_text += classification_report(y_test, lr_pred) + "\n"

# -----------------------------
# 6. Support Vector Machine
# -----------------------------

# Training Support Vector Machine (max-margin classifier)
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)
svm_acc = accuracy_score(y_test, svm_pred)

print("\nSVM Results")
print("Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred))

results_summary["SVM"] = svm_acc
results_text += "\nSVM Accuracy: " + str(svm_acc) + "\n"
results_text += classification_report(y_test, svm_pred) + "\n"

# -----------------------------
# 7. Save results to file
# -----------------------------

# Saving classification metrics to a text file for record
with open("results.txt", "w") as f:
    f.write(results_text)

print("Results saved to results.txt")

# -----------------------------
# 8. Plot comparison graph
# -----------------------------

# Plotting accuracy comparison of all models
models = list(results_summary.keys())
accuracies = list(results_summary.values())

plt.figure()
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison: Sports vs Politics Classification")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.002, str(round(v * 100, 2)) + "%", ha='center')

plt.ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)

# Saving comparison graph as image
plt.savefig("model_comparison.png")
print("Graph saved to model_comparison.png")
plt.show()
