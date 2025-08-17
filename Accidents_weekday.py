import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Accidents_cycling_csv.csv")

# Count accidents per weekday
weekday_counts = df['UWOCHENTAG'].value_counts().sort_index()

# Define order: Monday → Sunday
weekday_order = [2, 3, 4, 5, 6, 7, 1]
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Reorder counts
ordered_counts = [weekday_counts.get(day, 0) for day in weekday_order]

# Plot
fig, ax = plt.subplots()
ax.bar(weekday_labels, ordered_counts)

# Formatting
ax.set_title('Cycling Accidents per Weekday')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Accidents')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add commas to y-axis numbers
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()
