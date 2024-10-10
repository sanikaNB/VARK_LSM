import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
student_data = pd.read_csv('student_dataset_with_content.csv')

# Create a contingency table
contingency_table = pd.crosstab(student_data['Brain Dominance'], student_data['Primary VARK'])

# Perform the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output the Chi-squared statistic, p-value, and degrees of freedom
print(f"Chi-squared Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Interpretation of the p-value
if p < 0.05:
    print("There is a significant relationship between Brain Dominance and VARK types.")
else:
    print("No significant relationship was found between Brain Dominance and VARK types.")
