# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

# # Load the dataset
# student_data = pd.read_csv('D:\Workspace\Coding\Codes\Projects\VARK\student_dataset_with_content.csv')

# # 1. Handling missing values (if any)
# # Only fill missing values for numeric columns
# numeric_cols = student_data.select_dtypes(include=['float64', 'int64']).columns
# student_data[numeric_cols] = student_data[numeric_cols].fillna(student_data[numeric_cols].mean())

# # 2. Encoding categorical variables
# label_encoder = LabelEncoder()
# student_data['Gender'] = label_encoder.fit_transform(student_data['Gender'])  # Gender encoded
# student_data['Year'] = label_encoder.fit_transform(student_data['Year'])  # Year encoded
# student_data['Brain Dominance'] = label_encoder.fit_transform(student_data['Brain Dominance'])
# student_data['Primary VARK'] = label_encoder.fit_transform(student_data['Primary VARK'])

# # 3. Selecting Features (X) and Target (y)
# X = student_data.drop(columns=['Brain Dominance', 'Primary VARK', 'Student ID'])  # Features selected
# y_vark = student_data['Primary VARK']  # Target for VARK prediction
# y_brain = student_data['Brain Dominance']  # Target for Brain Dominance prediction

# # 4. Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)  # All features scaled

# # 5. Train-test split
# X_train, X_test, y_vark_train, y_vark_test = train_test_split(X_scaled, y_vark, test_size=0.2, random_state=42)
# X_train_brain, X_test_brain, y_brain_train, y_brain_test = train_test_split(X_scaled, y_brain, test_size=0.2, random_state=42)

# # Continue with the model training and evaluation as before...


# # 6. List of models to compare
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Random Forest': RandomForestClassifier(),
#     'SVM': SVC(),
#     'KNN': KNeighborsClassifier(),
#     'Neural Network (MLP)': MLPClassifier(max_iter=1000)
# }

# # 7. Function to train and evaluate models
# def evaluate_model(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     return accuracy, precision, recall, f1

# # 8. Evaluating models for VARK prediction
# print("VARK Prediction Model Comparison:\n")
# for name, model in models.items():
#     accuracy, precision, recall, f1 = evaluate_model(model, X_train, X_test, y_vark_train, y_vark_test)
#     print(f"{name}:\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

# # 9. Evaluating models for Brain Dominance prediction
# print("Brain Dominance Prediction Model Comparison:\n")
# for name, model in models.items():
#     accuracy, precision, recall, f1 = evaluate_model(model, X_train_brain, X_test_brain, y_brain_train, y_brain_test)
#     print(f"{name}:\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")













import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('D:\Workspace\Coding\Codes\Projects\VARK\student_dataset_vark_brain_mapped.csv')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Convert M/F to 0/1
df['Year'] = le.fit_transform(df['Year'])  # Convert year to categorical (1st, 2nd, etc.)
df['Brain Dominance'] = le.fit_transform(df['Brain Dominance'])  # Left/Right to 0/1
df['Primary VARK'] = le.fit_transform(df['Primary VARK'])
X = df.iloc[:, 6:-2]  # All columns except 'Primary VARK'
y = df['Primary VARK']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = []
performances = {}

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # Macro for multiclass
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    performances[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%\n")
    
    # Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    # print(f"Confusion Matrix for {model_name}:")
    # print(cm)
    
    # # Visualize Confusion Matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)  # Displaying the original VARK labels
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title(f'Confusion Matrix - {model_name}')
    # plt.show()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
evaluate_model(y_test, y_pred_lr, 'Logistic Regression')

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model(y_test, y_pred_dt, 'Decision Tree')

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
evaluate_model(y_test, y_pred_svm, 'SVM')

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, 'Random Forest')

# new_data = [[60, 60, 45, 48, 58, 28, 43, 46, 63, 52, 75, 62, 58, 63, 25.71, 21.31, 23.25, 29.74]]
# new_data_scaled = scaler.transform(new_data)  # Ensure to scale new data
# predicted_vark_rf = rf.predict(new_data_scaled)
# predicted_vark_lr = lr.predict(new_data_scaled)
# predicted_vark_svm = svm.predict(new_data_scaled)
# predicted_vark_dt = dt.predict(new_data_scaled)

# predicted_vark_label = le.inverse_transform(predicted_vark_rf)
# print(f"Predicted VARK type: {predicted_vark_label[0]}")

# predicted_vark_label = le.inverse_transform(predicted_vark_lr)
# print(f"Predicted VARK type: {predicted_vark_label[0]}")

# predicted_vark_label = le.inverse_transform(predicted_vark_svm)
# print(f"Predicted VARK type: {predicted_vark_label[0]}")

# predicted_vark_label = le.inverse_transform(predicted_vark_dt)
# print(f"Predicted VARK type: {predicted_vark_label[0]}")