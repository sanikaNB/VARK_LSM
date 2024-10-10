# main.py
from data_preprocessing import preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

def main():
    # Load and preprocess data
    file_path = 'D:\Workspace\Coding\Codes\Projects\VARK\student_dataset_vark_brain_mapped.csv'
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
