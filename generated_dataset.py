import pandas as pd
import numpy as np
import random

# Parameters
num_students = 2000
years = ["1st", "2nd", "3rd", "4th"]
genders = ["M", "F"]
brain_dominance_types = ["Left", "Right"]
vark_types = ['Visual', 'Aural', 'Read/Write', 'Kinesthetic']
content_types = ['Video', 'Audio', 'Article', 'Interactive']

# Function to generate brain dominance score based on a random scale
def generate_brain_dominance_score():
    score = random.randint(-15, 15)
    if score < 0:
        return "Left"
    else:
        return "Right"

# Function to generate correlated data
def generate_correlated_data(base, correlation, noise):
    return np.clip(base * correlation + np.random.normal(0, noise, len(base)), 0, 100)

# Sample data generation
data = {
    "Student ID": range(1, num_students + 1),
    "Age": [random.randint(18, 25) for _ in range(num_students)],
    "Gender": [random.choice(genders) for _ in range(num_students)],
    "Year": [random.choice(years) for _ in range(num_students)],
    "GPA": [round(random.uniform(2.0, 4.0), 2) for _ in range(num_students)],
    "Brain Dominance": [generate_brain_dominance_score() for _ in range(num_students)],
}

# Generate base scores for each VARK type
base_scores = {vark: np.random.uniform(60, 100, num_students) for vark in vark_types}

# Generate correlated data for each content type
for content_type, vark_type in zip(content_types, vark_types):
    data[f"Time Spent on {content_type} (min)"] = generate_correlated_data(base_scores[vark_type], 0.7, 10).astype(int)
    data[f"Avg {content_type} Score"] = generate_correlated_data(base_scores[vark_type], 0.8, 5).astype(int)
    data[f"{content_type} Completion Rate (%)"] = generate_correlated_data(base_scores[vark_type], 0.6, 10).astype(int)

# Additional data
data["Progress (%)"] = [random.randint(50, 100) for _ in range(num_students)]
data["Total Scroll Depth on Articles (%)"] = [random.randint(30, 100) for _ in range(num_students)]

# Create DataFrame
student_data = pd.DataFrame(data)

# Function to calculate VARK scores and assign primary type
def calculate_vark_scores_and_type(row):
    vark_scores = {}
    for vark_type, content_type in zip(vark_types, content_types):
        time = row[f'Time Spent on {content_type} (min)']
        score = row[f'Avg {content_type} Score']
        completion = row[f'{content_type} Completion Rate (%)']
        
        weighted_score = (time * 0.3) + (score * 0.4) + (completion * 0.3)
        vark_scores[vark_type] = weighted_score
    
    # Calculate percentage scores for each VARK type
    total_score = sum(vark_scores.values())
    vark_percentages = {k: round((v / total_score) * 100, 2) for k, v in vark_scores.items()}
    
    # Determine primary VARK type
    primary_vark = max(vark_percentages, key=vark_percentages.get)
    
    return pd.Series({**vark_percentages, 'Primary VARK': primary_vark})

# Calculate VARK scores and assign primary type
student_data[vark_types + ['Primary VARK']] = student_data.apply(calculate_vark_scores_and_type, axis=1)

# Map Brain Dominance to VARK Type
def map_vark_to_brain_dominance(row):
    if row['Brain Dominance'] == 'Left':
        if row['Primary VARK'] in ['Read/Write', 'Visual']:
            return 'Left'
        else:
            return 'Left'  # Default to Left for any other type
    else:  # Right Brain Dominant
        if row['Primary VARK'] in ['Aural', 'Kinesthetic']:
            return 'Right'
        else:
            return 'Right'  # Default to Right for any other type

# Apply the mapping function
student_data['Mapped VARK Type'] = student_data.apply(map_vark_to_brain_dominance, axis=1)

# Save to CSV
student_data.to_csv('student_dataset_vark_brain_mapped.csv', index=False)

print("Dataset generated and saved as 'student_dataset_vark_brain_mapped.csv'.")








#  import pandas as pd
# import numpy as np
# import random

# # parameters
# num_students = 3000
# years = ["1st", "2nd", "3rd", "4th"]
# genders = ["M", "F"]
# brain_dominance = ["Left", "Right", "Balanced"]
# vark_types = ['Visual', 'Aural', 'Read/Write', 'Kinesthetic']
# content_types = ['Video', 'Audio', 'Article', 'Interactive']

# # Function to generate correlated data
# def generate_correlated_data(base, correlation, noise):
#     return np.clip(base * correlation + np.random.normal(0, noise, len(base)), 0, 100)

# # Sample data generation
# data = {
#     "Student ID": range(1, num_students + 1),
#     "Age": [random.randint(18, 25) for _ in range(num_students)],
#     "Gender": [random.choice(genders) for _ in range(num_students)],
#     "Year": [random.choice(years) for _ in range(num_students)],
#     "GPA": [round(random.uniform(2.0, 4.0), 2) for _ in range(num_students)],
#     "Brain Dominance": [random.choice(brain_dominance) for _ in range(num_students)],
# }

# # Generate base scores for each VARK type
# base_scores = {vark: np.random.uniform(60, 100, num_students) for vark in vark_types}

# # Generate correlated data for each content type
# for content_type, vark_type in zip(content_types, vark_types):
#     data[f"Time Spent on {content_type} (min)"] = generate_correlated_data(base_scores[vark_type], 0.7, 10).astype(int)
#     data[f"Avg {content_type} Score"] = generate_correlated_data(base_scores[vark_type], 0.8, 5).astype(int)
#     data[f"{content_type} Completion Rate (%)"] = generate_correlated_data(base_scores[vark_type], 0.6, 10).astype(int)

# # Additional data
# data["Progress (%)"] = [random.randint(50, 100) for _ in range(num_students)]
# data["Total Scroll Depth on Articles (%)"] = [random.randint(30, 100) for _ in range(num_students)]

# # Create DataFrame
# student_data = pd.DataFrame(data)

# # Function to calculate VARK scores and assign primary type
# def calculate_vark_scores_and_type(row):
#     vark_scores = {}
#     for vark_type, content_type in zip(vark_types, content_types):
#         time = row[f'Time Spent on {content_type} (min)']
#         score = row[f'Avg {content_type} Score']
#         completion = row[f'{content_type} Completion Rate (%)']
        
#         weighted_score = (time * 0.3) + (score * 0.4) + (completion * 0.3)
#         vark_scores[vark_type] = weighted_score
    
#     # Calculate percentage scores for each VARK type
#     total_score = sum(vark_scores.values())
#     vark_percentages = {k: round((v / total_score) * 100, 2) for k, v in vark_scores.items()}
    
#     # Determine primary VARK type
#     primary_vark = max(vark_percentages, key=vark_percentages.get)
    
#     return pd.Series({**vark_percentages, 'Primary VARK': primary_vark})

# # Calculate VARK scores and assign primary type
# student_data[vark_types + ['Primary VARK']] = student_data.apply(calculate_vark_scores_and_type, axis=1)

# # Adjust primary VARK distribution to match typical populations
# # These percentages can be adjusted based on research or desired distribution
# target_distribution = {'Visual': 0.3, 'Aural': 0.25, 'Read/Write': 0.25, 'Kinesthetic': 0.2}
# current_distribution = student_data['Primary VARK'].value_counts(normalize=True)

# for vark_type, target_pct in target_distribution.items():
#     current_pct = current_distribution.get(vark_type, 0)
#     if current_pct < target_pct:
#         # Randomly select students to change to this VARK type
#         n_to_change = int((target_pct - current_pct) * len(student_data))
#         candidates = student_data[student_data['Primary VARK'] != vark_type].sample(n_to_change)
#         student_data.loc[candidates.index, 'Primary VARK'] = vark_type

# # Save to CSV
# student_data.to_csv('student_dataset_3000_for_training.csv', index=False)
# print("Dataset generated and saved as 'student_dataset_3000_for_training.csv'.")











# import pandas as pd
# import random

# # Parameters
# num_students = 3000
# years = ["1st", "2nd", "3rd", "4th"]
# genders = ["M", "F"]
# brain_dominance = ["Left", "Right", "Balanced"]

# # Sample data generation
# data = {
#     "Student ID": range(1, num_students + 1),
#     "Age": [random.randint(18, 25) for _ in range(num_students)],
#     "Gender": [random.choice(genders) for _ in range(num_students)],
#     "Year": [random.choice(years) for _ in range(num_students)],
#     "GPA": [round(random.uniform(2.0, 4.0), 2) for _ in range(num_students)],
#     "Time Spent on Videos (min)": [random.randint(60, 150) for _ in range(num_students)],
#     "Avg Video Score": [random.randint(70, 100) for _ in range(num_students)],
#     "Time Spent on Audio (min)": [random.randint(30, 120) for _ in range(num_students)],
#     "Avg Audio Score": [random.randint(60, 100) for _ in range(num_students)],
#     "Time Spent on Articles (min)": [random.randint(60, 120) for _ in range(num_students)],
#     "Avg Article Score": [random.randint(60, 100) for _ in range(num_students)],
#     "Time Spent on Kinesthetic (min)": [random.randint(30, 120) for _ in range(num_students)],
#     "Avg Kinesthetic Score": [random.randint(60, 100) for _ in range(num_students)],
#     "Progress (%)": [random.randint(50, 100) for _ in range(num_students)],
#     "Video Completion Rate (%)": [random.randint(70, 100) for _ in range(num_students)],
#     "Audio Completion Rate (%)": [random.randint(70, 100) for _ in range(num_students)],
#     "Article Completion Rate (%)": [random.randint(70, 100) for _ in range(num_students)],
#     "Kinesthetic Completion Rate (%)": [random.randint(70, 100) for _ in range(num_students)],
#     "Total Scroll Depth on Articles (%)": [random.randint(30, 100) for _ in range(num_students)],
#     # "Brain Dominance": [random.choice(brain_dominance) for _ in range(num_students)]
# }

# # Create DataFrame
# student_data = pd.DataFrame(data)

# # Function to assign VARK type considering Visual, Aural, Read/Write, and Kinesthetic
# def assign_vark(row):
#     video_time = row['Time Spent on Videos (min)']
#     audio_time = row['Time Spent on Audio (min)']
#     article_time = row['Time Spent on Articles (min)']
#     kinesthetic_time = row['Time Spent on Kinesthetic (min)']
#     video_score = row['Avg Video Score']
#     audio_score = row['Avg Audio Score']
#     article_score = row['Avg Article Score']
#     kinesthetic_score = row['Avg Kinesthetic Score']

    
#     max_time = max(video_time, audio_time, article_time, kinesthetic_time)
    
#     if max_time == video_time and video_score >= max(audio_score, article_score, kinesthetic_score):
#         return "Visual"
#     elif max_time == audio_time and audio_score >= max(video_score, article_score, kinesthetic_score):
#         return "Aural"
#     elif max_time == article_time and article_score >= max(video_score, audio_score, kinesthetic_score):
#         return "Read/Write"
#     elif max_time == kinesthetic_time and kinesthetic_score >= max(video_score, audio_score, article_score):
#         return "Kinesthetic"
#     else:
#         return "Mixed"

# # Apply the function to assign VARK types
# student_data['Predicted VARK'] = student_data.apply(assign_vark, axis=1)

# # Save to CSV
# student_data.to_csv('student_dataset_3000_with_all_vark_and_brain_dominance.csv', index=False)

# print("Dataset generated and saved as 'student_dataset_3000_with_all_vark_and_brain_dominance.csv'.")



