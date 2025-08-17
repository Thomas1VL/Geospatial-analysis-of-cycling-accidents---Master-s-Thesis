import pandas as pd
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv(r"D:\Unief\MA2\Masterproef\New_intersections_csv.csv")

#Count the number of intersections with and without accidents
accident_counts = df['accident?'].value_counts().sort_index()  # 0 first, then 1

#Print the amounts
print("Number of intersections without accidents (0):", accident_counts.get(0, 0))
print("Number of intersections with accidents (1):", accident_counts.get(1, 0))

#Plot
accident_counts.plot(kind='bar')

plt.title('Intersections With vs Without Cycling Accidents')
plt.xlabel('Accident Occurred? (0 = No, 1 = Yes)')
plt.ylabel('Number of Intersections')
plt.xticks([0, 1], ['No Cycling Accident', 'Cycling Accident'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()