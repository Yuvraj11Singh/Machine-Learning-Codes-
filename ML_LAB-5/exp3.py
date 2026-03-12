import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)

nb.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predictions
nb_pred = nb.predict(X_test)
dt_pred = dt.predict(X_test)

# Accuracy
nb_train_acc = nb.score(X_train, y_train)
nb_test_acc = accuracy_score(y_test, nb_pred)

dt_train_acc = dt.score(X_train, y_train)
dt_test_acc = accuracy_score(y_test, dt_pred)

# Bar Chart Comparison
models = ["Naive Bayes", "Decision Tree"]
train_acc = [nb_train_acc, dt_train_acc]
test_acc = [nb_test_acc, dt_test_acc]

x = np.arange(len(models))

plt.bar(x-0.2, train_acc, width=0.4, label="Train Accuracy")
plt.bar(x+0.2, test_acc, width=0.4, label="Test Accuracy")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.show()

# ROC Curve
nb_prob = nb.predict_proba(X_test)[:,1]
dt_prob = dt.predict_proba(X_test)[:,1]

fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)

plt.plot(fpr_nb, tpr_nb, label="Naive Bayes")
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Confusion Matrix Heatmaps
cm_nb = confusion_matrix(y_test, nb_pred)
cm_dt = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(cm_nb, annot=True, cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(cm_dt, annot=True, cmap="Greens")
plt.title("Decision Tree Confusion Matrix")

plt.show()