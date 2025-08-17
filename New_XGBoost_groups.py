import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# Load and prepare data
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

# Drop unwanted columns
columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']
df = df.drop(columns=columns_to_exclude, errors='ignore')

y = df['accident?']
X = df.drop(columns='accident?')
X = pd.get_dummies(X).fillna(0)

# Define your feature groups
feature_groups = {
    'Centrality': ['degree', 'closeness', 'betweennes', 'eigenvecto'],
    'Land Use': ['com%', 'ind%', 'res%'],
    'Traffic Elements': ['traffic li', 'crossing', 'stop sign', 'tram track'],
    'Cycling Infrastructure': ['sep_cyclew', 'non_sep_cy', 'cyclecross'],
    'Road': ['maxspeed_m', 'class nr_m'],
    'Nearby POI': ['Dis2School', 'Dis2PT']
}

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Define train/val/test split once
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_df, y, train_size=0.6, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Hyperparameter search with RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 5),
    'min_child_weight': randint(1, 10)
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=30,  # number of random parameter settings sampled
    scoring=make_scorer(f1_score),
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)
print("Best cross-validated F1 score:", random_search.best_score_)

# Train full model with best parameters
model_full = random_search.best_estimator_
f1_full = f1_score(y_val, model_full.predict(X_val))

# Store scores
f1_scores = {'All Features': f1_full}

# Train models leaving out each group
for group_name, group_features in feature_groups.items():
    drop_columns = [col for col in X.columns if any(col.startswith(f) for f in group_features)]
    X_reduced = X.drop(columns=drop_columns)
    X_reduced_scaled = scaler.fit_transform(X_reduced)
    X_reduced_df = pd.DataFrame(X_reduced_scaled, columns=X_reduced.columns)

    X_train_g, X_temp_g, y_train_g, y_temp_g = train_test_split(X_reduced_df, y, train_size=0.6, random_state=42, stratify=y)
    X_val_g, X_test_g, y_val_g, y_test_g = train_test_split(X_temp_g, y_temp_g, test_size=0.5, random_state=42, stratify=y_temp_g)

    # retrain using the best hyperparameters
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        **random_search.best_params_
    )
    model.fit(X_train_g, y_train_g)
    f1 = f1_score(y_val_g, model.predict(X_val_g))
    f1_scores[f'Without {group_name}'] = f1

# Plot results as vertical bar chart
delta_f1 = {k: f1_scores['All Features'] - v for k, v in f1_scores.items() if k != 'All Features'}
delta_f1_sorted = dict(sorted(delta_f1.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(10, 6))
bars = plt.bar(delta_f1_sorted.keys(), delta_f1_sorted.values())

plt.axhline(0, color='black', linewidth=1)
plt.ylabel('Δ F1 Score (compared to full model)')
plt.title('Impact of Removing Feature Groups on XGBoost')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

