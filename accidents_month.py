import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Load your dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Accidents_cycling_csv.csv")

# Count the number of accidents per month
# Assuming the column is named 'month'
monthly_counts = df['UMONAT'].value_counts().sort_index()  # Ensure months are in correct order

# Map numeric months to names (e.g., 1 → January)
month_names = [calendar.month_name[i] for i in monthly_counts.index]

# Create plot
fig, ax = plt.subplots()
ax.bar(month_names, monthly_counts.values)

# Formatting
ax.set_title('Cycling Accidents per Month')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Accidents')
ax.set_xticklabels(month_names, rotation=45, ha='right')  # Angle labels if needed
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Format y-axis with commas
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()
