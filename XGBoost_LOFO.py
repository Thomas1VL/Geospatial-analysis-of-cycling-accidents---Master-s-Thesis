import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import numpy as np

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

# Rename mapping for plot display
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

# Hyperparameter search with RandomizedSearchCV
param_dist = {
    "n_estimators": np.arange(100, 600, 100),
    "max_depth": np.arange(3, 11),
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "subsample": np.linspace(0.6, 1.0, 5),
    "colsample_bytree": np.linspace(0.6, 1.0, 5),
    "gamma": np.linspace(0, 0.5, 6),
    "reg_lambda": np.logspace(-1, 1, 5),  # L2
    "reg_alpha": np.logspace(-2, 1, 5),   # L1
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    param_distributions=param_dist,
    n_iter=30,  # number of random configs
    scoring=make_scorer(f1_score),
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)
print("Best CV F1 score:", random_search.best_score_)

# Train full model with best params
best_params = random_search.best_params_
model_full = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    **best_params
)
model_full.fit(X_train, y_train)
f1_full = f1_score(y_val, model_full.predict(X_val))

# LOFO Analysis
delta_f1 = {}
print("Running LOFO analysis with XGBoost (RandomizedSearchCV best params)...")

for feature in X.columns:
    X_reduced = X.drop(columns=feature)
    X_reduced_scaled = scaler.fit_transform(X_reduced)
    X_reduced_df = pd.DataFrame(X_reduced_scaled, columns=X_reduced.columns)

    # Resplit
    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
        X_reduced_df, y, train_size=0.6, random_state=42, stratify=y
    )
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
        X_temp_r, y_temp_r, test_size=0.5, random_state=42, stratify=y_temp_r
    )

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        **best_params
    )
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
plt.title('Leave-One-Feature-Out Impact on XGBoost')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
