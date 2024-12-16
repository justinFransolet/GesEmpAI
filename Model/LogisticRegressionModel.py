import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de régression logistique et évalue ses performances.

    :param X_train: Les données d'entraînement
    :param y_train: Les labels d'entraînement
    :param X_test: Les données de test
    :param y_test: Les labels de test
    :return: Le modèle entraîné
    """
    logreg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train)
    
    # Prédictions
    y_pred = logreg.predict(X_test)
    y_proba = logreg.predict_proba(X_test)[:, 1]

    # Évaluation
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.2f}")

    # Courbe ROC
    RocCurveDisplay.from_estimator(logreg, X_test, y_test)
    
    return logreg


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    """
    Ajuste les hyperparamètres de la régression logistique à l'aide de GridSearchCV.

    :param X_train: Les données d'entraînement
    :param y_train: Les labels d'entraînement
    :param X_test: Les données de test
    :param y_test: Les labels de test
    :return: Le meilleur modèle entraîné
    """
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'max_iter': [500, 1000]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=42),
        param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    print("\nMeilleurs paramètres trouvés :")
    print(grid_search.best_params_)

    # Évaluer le modèle optimisé
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nRapport de classification (après tuning) :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC (après tuning) : {roc_auc_score(y_test, y_proba):.2f}")

    # Courbe ROC
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    
    return best_model
