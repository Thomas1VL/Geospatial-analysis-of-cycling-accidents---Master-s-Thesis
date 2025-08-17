import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\New_intersections_csv.csv")

# Separate target and features
y = df['accident?']
X = df.drop(columns='accident?')

# Combine into one dataframe again to subset easily
data = pd.concat([X, y], axis=1)

# Separate classes
accidents = data[data['accident?'] == 1]
no_accidents = data[data['accident?'] == 0]

# Undersample no-accidents to match the number of accidents
no_accidents_sampled = no_accidents.sample(n=len(accidents), random_state=42)

# Combine and shuffle
balanced_data = pd.concat([accidents, no_accidents_sampled])
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Export to CSV
balanced_data.to_csv(r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv", index=False)
