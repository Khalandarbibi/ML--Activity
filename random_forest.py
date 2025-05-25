# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('diabetes_subset_25k.csv')
df.columns = df.columns.str.strip()  # Remove trailing spaces in column names

# Target column
target_column = 'readmitted'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found.")

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Factorize target if it's categorical
if y.dtype == 'object':
   y, label_names = pd.factorize(y)
   y = pd.Series(y, index=df.index)  # Re-wrap as pandas Series


# Handle invalid/misplaced values
X = X.replace('?', np.nan)
X = X.dropna(axis=1, how='any')  # Drop columns with missing values (optional: you can impute too)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Safety check: ensure no string data is in numerical columns
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')  # convert any hidden strings to NaN
X = X.dropna(axis=0)  # Drop rows with NaNs (after conversion)

# Final column lists after cleanup
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Split after cleaning
X_train, X_test, y_train, y_test = train_test_split(
    X, y.loc[X.index], test_size=0.2, random_state=42, stratify=y.loc[X.index]
)

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# SVC pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Hyperparameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto']
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model and predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Accuracy and classification report
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ SVM Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Actual vs Predicted plot
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
plt.xticks(x, [label_names[i] if 'label_names' in locals() else str(i) for i in all_labels])
plt.legend()
plt.tight_layout()
plt.show()
