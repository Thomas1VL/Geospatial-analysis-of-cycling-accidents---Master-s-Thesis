import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Accidents_cycling_csv.csv")

# Count accidents per hour
hourly_counts = df['USTUNDE'].value_counts().sort_index()

# Plot
fig, ax = plt.subplots()
ax.bar(hourly_counts.index.astype(str), hourly_counts.values)

# Formatting
ax.set_title('Cycling Accidents per Hour of the Day')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Number of Accidents')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add commas to y-axis numbers
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()
