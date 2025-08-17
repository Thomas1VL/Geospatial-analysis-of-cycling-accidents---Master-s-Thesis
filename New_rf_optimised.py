import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load already balanced dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

# Drop unwanted columns
columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']
df = df.drop(columns=columns_to_exclude, errors='ignore')

# Separate target and features
y = df['accident?']
X = df.drop(columns='accident?')

# One-hot encode categorical variables and fill missing values
X = pd.get_dummies(X)
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 60% train / 20% val / 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, train_size=0.6, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# RandomizedSearchCV setup
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# Evaluate on validation set
best_rf = random_search.best_estimator_
y_val_pred = best_rf.predict(X_val)
print("Best Parameters:", random_search.best_params_)
print("Validation Set Evaluation")
print("F1 Score (accident class):", round(f1_score(y_val, y_val_pred, pos_label=1), 3))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=0))

# Final evaluation on test set
y_test_pred = best_rf.predict(X_test)
print("\nTest Set Evaluation")
print("F1 Score (accident class):", round(f1_score(y_test, y_test_pred, pos_label=1), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

# Feature importance
importances = pd.Series(best_rf.feature_importances_, index=X.columns)

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

importances_display = importances.copy()
importances_display.index = [rename_dict.get(col, col) for col in importances_display.index]
importances_sorted = importances_display.sort_values(ascending=False)

print("Feature Importance:")
print(importances_sorted.head(20))

# Plot top features with grid
plt.figure(figsize=(8, 5))
importances_sorted.head(20).plot(kind='barh', title='Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
