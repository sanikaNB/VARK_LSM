import random
import pandas as pd

# Define the number of students and the number of questions
num_students = 2000
num_questions = 15  # Based on the number of questions in your original questionnaire

# Function to generate random responses ('a', 'b', 'c') for each student
def generate_responses(num_questions):
    return [random.choice(['a', 'b', 'c']) for _ in range(num_questions)]

# Function to calculate brain dominance score based on 'a' and 'b' answers
def calculate_brain_dominance(responses):
    a_count = responses.count('a')  # Count 'a' responses
    b_count = responses.count('b')  # Count 'b' responses
    score = b_count - a_count        # Calculate the score

    # Determine brain dominance based only on left and right
    if score < 0:
        return score, "Left Brain Dominant"
    else:
        return score, "Right Brain Dominant"

# Generate the dataset for 2000 students
students_data = []
for student_id in range(1, num_students + 1):
    responses = generate_responses(num_questions)  # Generate random responses
    score, dominance = calculate_brain_dominance(responses)  # Calculate score and dominance
    students_data.append({
        'Student ID': student_id,
        'Responses': responses,
        'Score': score,
        'Brain Dominance': dominance
    })

# Convert the data to a pandas DataFrame
df = pd.DataFrame(students_data)

# Save the DataFrame to a CSV file
df.to_csv('brain_dominance_left_right_dataset.csv', index=False)

print("Dataset generated and saved as 'brain_dominance_left_right_dataset.csv'.")
