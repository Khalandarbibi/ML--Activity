import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_csv('diabetes_subset_25k.csv')
df.columns = df.columns.str.strip()

# Sample smaller dataset for faster processing
df = df.sample(1000, random_state=42).reset_index(drop=True)

# Target variable
target_column = 'readmitted'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Pipeline
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Reduced hyperparameter grid for faster tuning
param_grid = {
    'classifier__n_neighbors': [3, 5, 7],
    'classifier__weights': ['uniform'],
    'classifier__metric': ['euclidean']
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV with parallel jobs
grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit grid search
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)
print("Grid Search complete!")

print("Best Parameters:", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\nKNN Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 📊 VISUALIZATIONS
# -----------------------------

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. Prediction Count (Bar Plot)
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred, palette='Set2')
plt.title('Predicted Class Distribution')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.show()

# 3. Actual Class Proportions (Pie Chart)
labels = y.unique()
sizes = [sum(y_test == label) for label in labels]

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title("Actual Class Distribution")
plt.axis('equal')
plt.show()

# 4. Feature Importance Proxy (Variance)
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat']
encoded_cols = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cols)

X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
X_dense = X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed

variances = np.var(X_dense, axis=0)
top_idx = np.argsort(variances)[-10:]
top_features = [all_feature_names[i] for i in top_idx]
top_values = variances[top_idx]

plt.figure(figsize=(10,5))
sns.barplot(x=top_values, y=top_features, palette='coolwarm')
plt.title("Top 10 Influential Features (by Variance)")
plt.xlabel("Variance")
plt.ylabel("Features")
plt.show()

# 5. Decision Boundary (only 2 numerical features)
features_2d = ['time_in_hospital', 'num_medications']
if all(f in numerical_cols for f in features_2d):
    X_vis = df[features_2d]
    y_vis = df[target_column]
    y_vis_encoded = pd.factorize(y_vis)[0]

    scaler_vis = StandardScaler()
    X_vis_scaled = scaler_vis.fit_transform(X_vis)

    knn_vis = KNeighborsClassifier(n_neighbors=grid_search.best_params_['classifier__n_neighbors'])
    knn_vis.fit(X_vis_scaled, y_vis_encoded)

    x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
    y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
    plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis_encoded, edgecolors='k', cmap=cmap)
    plt.title("KNN Decision Boundary (2 Features)")
    plt.xlabel(features_2d[0])
    plt.ylabel(features_2d[1])
    plt.show()
else:
    print("Features for decision boundary not found in numerical columns.")
