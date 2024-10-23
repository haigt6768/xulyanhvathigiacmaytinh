import numpy as np
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load dataset (Digits dataset từ sklearn)
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 2. Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm tính toán và in các kết quả
def evaluate_model(model, X_train, X_test, y_train, y_test, algo_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')

    print(f"--- {algo_name} ---")
    print(f"Time: {elapsed_time:.4f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print()

# 3. SVM
svm_model = SVC()
evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM")

# 4. KNN
knn_model = KNeighborsClassifier()
evaluate_model(knn_model, X_train, X_test, y_train, y_test, "KNN")

# 5. Decision Tree
tree_model = DecisionTreeClassifier()
evaluate_model(tree_model, X_train, X_test, y_train, y_test, "Decision Tree")
