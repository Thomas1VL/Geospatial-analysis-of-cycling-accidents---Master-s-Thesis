import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

# Drop unwanted columns
columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']
df = df.drop(columns=columns_to_exclude, errors='ignore')

# Separate target and features
y = df['accident?']
X = df.drop(columns='accident?')
X = pd.get_dummies(X).fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
# -> 60% train, 20% val, 20% test

# RandomizedSearchCV for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit model
random_search.fit(X_train, y_train)

# Evaluate on validation set
best_xgb = random_search.best_estimator_
y_val_pred = best_xgb.predict(X_val)

print("Best Parameters:", random_search.best_params_)
print("Validation Set Evaluation")
print("F1 Score (accident class):", round(f1_score(y_val, y_val_pred, pos_label=1), 3))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=0))

# Test set performance
y_test_pred = best_xgb.predict(X_test)

print("\nTest Set Evaluation")
print("F1 Score (accident class):", round(f1_score(y_test, y_test_pred, pos_label=1), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

# Feature importance
importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
importances_sorted = importances.abs().sort_values(ascending=False).head(20)

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

importances_sorted.index = [rename_dict.get(feat, feat) for feat in importances_sorted.index]

print("Feature Importance:")
print(importances_sorted.sort_values(ascending=False).head(20))

# Plot
plt.figure(figsize=(9, 5))
importances_sorted.sort_values(ascending=False).plot(kind='barh', title='Feature Importance (XGBoost)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
