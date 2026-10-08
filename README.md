# Task 4: Classification with Logistic Regression

## Objective

Build a binary classifier using Logistic Regression and evaluate it using classification metrics.

## Dataset

The Breast Cancer Wisconsin Dataset provided through scikit-learn is used. It contains measurements computed from digitized breast mass images and a binary target indicating the class.

The dataset is loaded directly with `sklearn.datasets.load_breast_cancer()`, so no separate dataset download is required.

## Tools

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Workflow

1. Load the Breast Cancer Wisconsin Dataset.
2. Separate features and target.
3. Split the data into training and testing sets.
4. Standardize the features.
5. Train a Logistic Regression classifier.
6. Evaluate accuracy, precision, recall, confusion matrix, and ROC-AUC.
7. Plot the confusion matrix and ROC curve.
8. Test different classification thresholds.
9. Plot the sigmoid function.

## How to Run

Install the required packages:

```bash
pip install pandas scikit-learn matplotlib
```

Run:

```bash
python logistic_regression.py
```

The program prints the evaluation metrics and threshold comparison and generates:

- `confusion_matrix.png`
- `roc_curve.png`
- `sigmoid_curve.png`

## Key Concepts

### Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of a class. It applies the sigmoid function to a linear combination of input features.

### Sigmoid Function

The sigmoid function is:

`sigmoid(z) = 1 / (1 + e^(-z))`

It converts a real-valued input into a value between 0 and 1, which can be interpreted as a probability.

### Threshold

The default classification threshold is 0.50. A different threshold can be selected depending on whether precision, recall, or another evaluation objective is more important.

## Task Requirement Coverage

- Binary classification dataset
- Train/test split
- Feature standardization
- Logistic Regression
- Confusion matrix
- Precision
- Recall
- ROC-AUC
- Threshold tuning
- Sigmoid function
