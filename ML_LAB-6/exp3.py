import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(probability=True)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predictions
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)

# Accuracy
knn_train = knn.score(X_train, y_train)
knn_test = accuracy_score(y_test, knn_pred)

svm_train = svm.score(X_train, y_train)
svm_test = accuracy_score(y_test, svm_pred)

# Bar Chart Comparison
models = ["KNN", "SVM"]
train_acc = [knn_train, svm_train]
test_acc = [knn_test, svm_test]

x = np.arange(len(models))

plt.bar(x-0.2, train_acc, width=0.4, label="Train Accuracy")
plt.bar(x+0.2, test_acc, width=0.4, label="Test Accuracy")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.show()

# ROC Curve
knn_prob = knn.predict_proba(X_test)[:,1]
svm_prob = svm.predict_proba(X_test)[:,1]

fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)

plt.plot(fpr_knn, tpr_knn, label="KNN")
plt.plot(fpr_svm, tpr_svm, label="SVM")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Confusion Matrix Heatmaps
cm_knn = confusion_matrix(y_test, knn_pred)
cm_svm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(cm_knn, annot=True, cmap="Blues")
plt.title("KNN Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(cm_svm, annot=True, cmap="Greens")
plt.title("SVM Confusion Matrix")

plt.show()