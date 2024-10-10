# model_evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"{model_name} Metrics:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%\n")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        
        # Save confusion matrix plot as a file
        plt.figure(figsize=(10, 6))  # Optional: specify the figure size
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f'confusion_matrix_{model_name}.png', bbox_inches='tight')  # Save the figure
        plt.close()  # Close the plot to avoid display in non-interactive environments
