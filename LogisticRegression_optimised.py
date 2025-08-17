import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load already balanced dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

# Drop unwanted columns
columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']
df = df.drop(columns=columns_to_exclude, errors='ignore')

# Separate target and features
y = df['accident?']
X = df.drop(columns='accident?')

# One-hot encode and handle missing values
X = pd.get_dummies(X).fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Validation/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, train_size=0.6, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# GridSearchCV for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # Only solver that supports both L1 and L2
}

log_reg = LogisticRegression(max_iter=1000, random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_log_reg = grid_search.best_estimator_

print("Best Parameters Found:", grid_search.best_params_)

# Evaluate on validation set
y_val_pred = best_log_reg.predict(X_val)

print("\nValidation Set Evaluation")
print("F1 Score:", round(f1_score(y_val, y_val_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=0))

# Final evaluation on test set
y_test_pred = best_log_reg.predict(X_test)

print("\nTest Set Evaluation")
print("F1 Score:", round(f1_score(y_test, y_test_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

# Feature importance (log odds)
coefficients = pd.Series(best_log_reg.coef_[0], index=X.columns)

# Rename variables for display
rename_dict = {
    'degree': 'Degree centrality',
    'closeness': 'Closeness centrality',
    'betweennes': 'Betweenness centrality',
    'eigenvecto': 'Alpha centrality',
    'com%': '% of commercial land use',
    'ind%': '% of industrial land use',
    'res%': '% of residential land use',
    'traffic li': 'Traffic light',
    'crossing': 'Zebra crossing',
    'stop sign': 'Stop sign',
    'tram track': 'Tram track',
    'sep_cyclew': 'Separated cycleway',
    'non_sep_cy': 'Non-separated cycleway',
    'cyclecross': 'Cycle crossing',
    'maxspeed_m': 'Highest speed limit',
    'class nr_m': 'Road classification',
    'Dis2School': 'Distance to nearest school',
    'Dis2PT': 'Distance to nearest public transport stop',
    'accident?': 'Accident?'
}

coefficients_display = coefficients.copy()
coefficients_display.index = [rename_dict.get(col, col) for col in coefficients_display.index]
coefficients_sorted = coefficients_display.sort_values(ascending=False)

print("Top Feature Importance:")
print(coefficients_sorted.head(20))

# Plot top features
plt.figure(figsize=(8, 5))
coefficients_sorted.head(20).plot(kind='barh', title='Feature Importance (Logistic Regression)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Vertical gridlines
plt.tight_layout()
plt.show()
