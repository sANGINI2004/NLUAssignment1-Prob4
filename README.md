# NLU Assignment - Problem 04 : Sports vs Politics Text Classification


## Objective

The goal of this assignment is to build a machine learning based text classifier that categorizes news articles into two topics:

* **SPORT**
* **POLITICS**

The project evaluates how well traditional machine learning algorithms perform on textual data using statistical feature representations.

---

## Dataset

The dataset used is the **20 Newsgroups Dataset**, accessed directly using the Scikit-learn API.

Selected categories:

* `rec.sport.baseball`
* `rec.sport.hockey`
* `talk.politics.mideast`
* `talk.politics.misc`

These were merged into two labels:

| Category        | Final Class |
| --------------- | ----------- |
| Sports groups   | SPORT       |
| Politics groups | POLITICS    |

Total documents used: **3708**

Dataset is downloaded automatically when the program runs.

---

## Feature Extraction

Text documents are converted into numerical vectors using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

Configuration:

* Unigram + Bigram features
* English stop-words removed
* Minimum document frequency = 3

This helps capture contextual phrases like *"white house"* instead of only single words.

---

## Models Implemented

The following supervised learning algorithms were compared:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)

---

## Results Summary

| Model               | Accuracy    |
| ------------------- | ----------- |
| Naive Bayes         | ~95%        |
| Logistic Regression | ~94%        |
| SVM                 | ~95% (Best) |

SVM achieved the highest performance due to effective handling of high-dimensional sparse text features.

---

## How to Run

### 1. Install dependencies

```
pip install scikit-learn matplotlib numpy
```

### 2. Execute the program

```
python3 code/B22CS045_prob4.py
```

The dataset will automatically download and the program will:

* Train models
* Evaluate performance
* Print metrics
* Save comparison graph

---

## Repository Structure

```
code/        → Python implementation
report/      → Assignment report (PDF)
results/     → Output and generated plots
README.md    → Project description
```

---

## Output

The program generates:

* Accuracy, Precision, Recall, F1-Score for each model
* `model_comparison.png` graph
* Sample output stored in `results/sample_output.txt`

---

## Conclusion

Traditional machine learning methods remain highly effective for text classification. With simple TF-IDF features, all models achieved above 94% accuracy, with SVM performing best.

---

## Reproducibility

No manual dataset download is required. Running the script once reproduces the complete experiment.
