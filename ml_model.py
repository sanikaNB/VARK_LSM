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

df = pd.read_csv('D:\Workspace\Coding\Codes\Projects\VARK\VARK_LMS\student_dataset_3000_for_training.csv')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Convert M/F to 0/1
df['Year'] = le.fit_transform(df['Year'])  # Convert year to categorical (1st, 2nd, etc.)
df['Brain Dominance'] = le.fit_transform(df['Brain Dominance'])  # Left/Right to 0/1
df['Primary VARK'] = le.fit_transform(df['Primary VARK'])
X = df.iloc[:, 6:-1]  # All columns except 'Primary VARK'
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

new_data = [[60, 60, 45, 48, 58, 28, 43, 46, 63, 52, 75, 62, 58, 63, 25.71, 21.31, 23.25, 29.74]]
new_data_scaled = scaler.transform(new_data)  # Ensure to scale new data
predicted_vark_rf = rf.predict(new_data_scaled)
predicted_vark_lr = lr.predict(new_data_scaled)
predicted_vark_svm = svm.predict(new_data_scaled)
predicted_vark_dt = dt.predict(new_data_scaled)

predicted_vark_label = le.inverse_transform(predicted_vark_rf)
print(f"Predicted VARK type: {predicted_vark_label[0]}")

predicted_vark_label = le.inverse_transform(predicted_vark_lr)
print(f"Predicted VARK type: {predicted_vark_label[0]}")

predicted_vark_label = le.inverse_transform(predicted_vark_svm)
print(f"Predicted VARK type: {predicted_vark_label[0]}")

predicted_vark_label = le.inverse_transform(predicted_vark_dt)
print(f"Predicted VARK type: {predicted_vark_label[0]}")