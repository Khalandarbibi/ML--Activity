import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn

# Show sklearn version
print("✅ scikit-learn version:", sklearn.__version__)

# Load dataset
df = pd.read_csv('diabetes_subset_25k.csv')
df.columns = df.columns.str.strip()

# Target column
target_column = 'readmitted'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found.")

X = df.drop(target_column, axis=1)
y = df[target_column]

# Factorize target if categorical
if y.dtype == 'object':
    y, label_names = pd.factorize(y)
    label_mapping = dict(enumerate(label_names))
else:
    label_names = [str(i) for i in sorted(np.unique(y))]

# Separate feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Handle OneHotEncoder compatibility with different sklearn versions
from sklearn import __version__ as sklearn_version
from packaging import version

if version.parse(sklearn_version) >= version.parse("1.2"):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', encoder, categorical_cols)
])

# SVM pipeline
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Parameter grid
param_grid = {
    'classifier__C': [1],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale']
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# GridSearchCV
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)

# Predictions
y_pred = best_svm.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ SVM Accuracy: {acc*100:.2f}%")

print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=label_names, yticklabels=label_names)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Actual vs Predicted Distribution
plt.figure(figsize=(8, 5))
all_labels = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
actual_counts = pd.Series(y_test).value_counts().reindex(all_labels, fill_value=0)
pred_counts = pd.Series(y_pred).value_counts().reindex(all_labels, fill_value=0)
x = np.arange(len(all_labels))
width = 0.35

plt.bar(x - width/2, actual_counts.values, width, label='Actual', color='skyblue')
plt.bar(x + width/2, pred_counts.values, width, label='Predicted', color='lightcoral')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Actual vs Predicted Class Distribution')
plt.xticks(x, [label_names[i] for i in all_labels])
plt.legend()
plt.tight_layout()
plt.show()
