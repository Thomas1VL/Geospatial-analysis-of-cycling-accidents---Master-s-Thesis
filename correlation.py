import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv")

columns_to_exclude = ['osm_id', 'code', 'fclass', 'name', 'ref', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel',
                      'road_class', 'unique_id', 'PTtype']  # Columns to remove
include_target_in_corr = True  # Set False if you don't want to include 'accident?' column

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


# Drop unwanted columns
df_corr = df.drop(columns=columns_to_exclude, errors='ignore')

# remove the target column (accident?)
# if not include_target_in_corr:
#   df_corr = df_corr.drop(columns=['accident?'], errors='ignore')

# Fill NaNs with 0
df_corr = df_corr.fillna(0)

# Select only numeric columns
df_corr = df_corr.select_dtypes(include='number')

# Correlation matrix
corr_matrix = df_corr.corr(method='pearson')

# Move 'accident?' column to the end before renaming
accident_col = 'accident?'
if accident_col in corr_matrix.columns:
    cols = [c for c in corr_matrix.columns if c != accident_col] + [accident_col]
    corr_matrix = corr_matrix.loc[cols, cols]

# Rename columns for display
display_names = [rename_dict.get(col, col) for col in corr_matrix.columns]
corr_matrix.index = display_names
corr_matrix.columns = display_names

# Visualisation
plt.figure(figsize=(13, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.75})
plt.title("Pearson Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
