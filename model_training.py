# model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier()
    }

    # Hyperparameter tuning for Random Forest as an example
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search_rf = GridSearchCV(estimator=models['Random Forest'], param_grid=rf_param_grid, cv=5)
    grid_search_rf.fit(X_train, y_train)
    models['Random Forest'] = grid_search_rf.best_estimator_  # Update the model with the best parameters

    # Fit all models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{model_name} trained successfully!")

    return models
