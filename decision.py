# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes_subset_25k.csv')
df.columns = df.columns.str.strip()

# Features and target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# One-hot encode categorical variables (customize if more exist)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Encode target if needed
if y.dtype == 'object':
    y, class_names = pd.factorize(y)
    class_names = class_names.tolist()  # Convert Index to list
else:
    class_names = [str(cls) for cls in sorted(np.unique(y))]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree & GridSearchCV
dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_dt = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)

# Predict
y_pred = best_dt.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"✅ Decision Tree Accuracy: {acc * 100:.2f}%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ✅ Decision Tree Graph
plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    filled=True,
    feature_names=X.columns,
    class_names=class_names,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()
