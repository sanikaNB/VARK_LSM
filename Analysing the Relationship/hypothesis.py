import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:\Workspace\Coding\Codes\Projects\VARK\student_dataset_vark_brain_mapped.csv')

# Contingency table to show counts of VARK types by Brain Dominance
contingency_table = pd.crosstab(df['Brain Dominance'], df['Primary VARK'])
print(contingency_table)

# Visualizing the relationship between Brain Dominance and VARK types
sns.countplot(data=df, x='Primary VARK', hue='Brain Dominance')
plt.title('Distribution of VARK Types by Brain Dominance')
plt.show()
