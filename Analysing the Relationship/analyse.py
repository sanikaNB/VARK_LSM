import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
student_data = pd.read_csv('D:\Workspace\Coding\Codes\Projects\VARK\student_dataset_vark_brain_mapped.csv')

# Create a contingency table to show counts of VARK types by Brain Dominance
contingency_table = pd.crosstab(student_data['Brain Dominance'], student_data['Primary VARK'])

# Create a bar plot to visualize the counts (number of students)
fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size

# Plot the stacked bar chart with the number of students
bars = contingency_table.plot(kind='bar', stacked=True, ax=ax)

# Annotate the number of students on each section of the stacked bar
for c in bars.containers:
    # Add labels for the number of students in each section
    bars.bar_label(c, label_type='center', fmt='%d', color='white', fontsize=10)

print(student_data[['Brain Dominance', 'Primary VARK']].isnull().sum())


# Set titles and labels
plt.title('VARK Types by Brain Dominance (Number of Students)')
plt.xlabel('Brain Dominance')
plt.ylabel('Number of Students')
plt.legend(title='Primary VARK Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('vark_types_by_brain_dominance_number_of_students.png', bbox_inches='tight')

# Show the plot
plt.show()

print(len(student_data))
