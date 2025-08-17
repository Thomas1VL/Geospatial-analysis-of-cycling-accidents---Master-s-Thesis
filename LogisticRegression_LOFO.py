import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer

# Load and prepare data
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

# Drop unwanted columns
columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']
df = df.drop(columns=columns_to_exclude, errors='ignore')

# Separate target and features
y = df['accident?']
X = df.drop(columns='accident?')
X = pd.get_dummies(X).fillna(0)

# Rename mapping for display
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
    'Dis2PT': 'Distance to nearest PT stop',
    'accident?': 'Accident?'
}

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train/Validation/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df, y, train_size=0.6, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Hyperparameter search with GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],  # saga supports all penalties
    'l1_ratio': [0, 0.5, 1]  # only used for elasticnet
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=5000, random_state=42),
    param_grid,
    scoring=make_scorer(f1_score),
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validated F1 score:", grid_search.best_score_)

# Full model with best params
best_params = grid_search.best_params_
full_model = LogisticRegression(max_iter=5000, random_state=42, **best_params)
full_model.fit(X_train, y_train)
f1_full = f1_score(y_val, full_model.predict(X_val))

# LOFO
delta_f1 = {}

print("Running LOFO analysis for Logistic Regression with GridSearchCV...")

for feature in X.columns:
    X_reduced = X.drop(columns=feature)
    X_reduced_scaled = scaler.fit_transform(X_reduced)
    X_reduced_df = pd.DataFrame(X_reduced_scaled, columns=X_reduced.columns)

    # Split
    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(X_reduced_df, y, train_size=0.6, random_state=42, stratify=y)
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(X_temp_r, y_temp_r, test_size=0.5, random_state=42, stratify=y_temp_r)

    model = LogisticRegression(max_iter=5000, random_state=42, **best_params)
    model.fit(X_train_r, y_train_r)
    f1_reduced = f1_score(y_val_r, model.predict(X_val_r))

    delta = f1_full - f1_reduced
    delta_f1[feature] = delta

# Sort and prepare display labels
delta_f1_sorted = dict(sorted(delta_f1.items(), key=lambda x: x[1], reverse=True))
display_labels = [rename_dict.get(feat, feat) for feat in delta_f1_sorted.keys()]

# Plot
plt.figure(figsize=(10, max(6, len(display_labels) * 0.35)))
plt.barh(display_labels, delta_f1_sorted.values())
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Δ F1 Score (compared to full model)')
plt.title('Leave-One-Feature-Out Impact on Logistic Regression')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
