import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\Accidents_all_csv.csv")

#Count the number of intersections with and without accidents
accident_counts = df['IstRad'].value_counts().sort_index()  # 0 first, then 1

#Print the amounts
print("Number of accidents without cyclists (0):", accident_counts.get(0, 0))
print("Number of accidents involving cyclists (1):", accident_counts.get(1, 0))

#Plot
fig, ax = plt.subplots()
accident_counts.plot(kind='bar', ax=ax)

ax.set_title('Accidents With vs Without Cyclists')
ax.set_xlabel('Cyclist involved? (0 = No, 1 = Yes)')
ax.set_ylabel('Number of Accidents')
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Cyclist involved', 'Cyclist involved'], rotation = 0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Format the y-axis to show plain integers instead of the  scientific notation
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.show()