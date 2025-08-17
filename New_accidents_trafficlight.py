import pandas as pd
import matplotlib.pyplot as plt

def plot_tram_accident_counts(csv_path, title):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Make sure 'tram track' is binary (0 = no, 1 = yes)
    df['traffic li'] = df['traffic li'].astype(int)

    # Group and count
    group_counts = df.groupby(['accident?', 'traffic li']).size().reset_index(name='count')

    # Pivot for plotting
    pivot_df = group_counts.pivot(index='accident?', columns='traffic li', values='count').fillna(0)
    pivot_df.columns = ['No Traffic light', 'Traffic light']

    # Plot grouped bar chart
    pivot_df.plot(kind='bar', figsize=(8, 5))
    plt.title(title)
    plt.xlabel('Accident? (0 = No, 1 = Yes)')
    plt.ylabel('Number of Intersections')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ---- Run for both datasets ----
plot_tram_accident_counts(
    r"D:\Unief\MA2\Masterproef\New_intersections_csv.csv",
    "Intersections with and without Traffic Lights (Full dataset)"
)

plot_tram_accident_counts(
    r"D:\Unief\MA2\Masterproef\Balanced_intersections.csv",
    "Intersections with and without Traffic Lights (Balanced dataset)"
)

