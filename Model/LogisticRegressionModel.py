from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV


def ensure_column_match(X_train, X_test):
    """
    Vérifie que les colonnes des ensembles d'entraînement et de test correspondent.
    Supprime les colonnes supplémentaires si nécessaire.

    :param X_train: Les données d'entraînement
    :param X_test: Les données de test
    :return: Les ensembles d'entraînement et de test mis à jour
    """
    common_columns = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_columns]
    X_test = X_test[common_columns]
    return X_train, X_test

def train_logistic_model(X_train, y_train, model=None):
    """
    Entraîne un modèle de régression logistique sur les données d'entraînement.

    :param X_train: Les données d'entraînement
    :param y_train: Les labels d'entraînement
    :param model: Le modèle à entraîner (par défaut, LogisticRegression)
    :return: Le modèle de régression logistique entraîné
    """
    if model is None:
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    return model


def predict_logistic_model(model, X_test):
    """
    Effectue des prédictions avec un modèle de régression logistique.

    :param model: Le modèle de régression logistique entraîné
    :param X_test: Les données de test
    :return: Les prédictions et les probabilités pour la classe positive
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate_logistic_model(y_test, y_pred, y_proba):
    """
    Évalue les performances d'un modèle de régression logistique.

    :param y_test: Les labels réels de test
    :param y_pred: Les prédictions du modèle
    :param y_proba: Les probabilités pour la classe positive
    """
    print("\n=== Évaluation du modèle de régression logistique ===")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.5f}")


def display_logistic_roc_curve(model, X_test, y_test):
    """
    Affiche la courbe ROC pour un modèle de régression logistique.

    :param model: Le modèle de régression logistique entraîné
    :param X_test: Les données de test
    :param y_test: Les labels réels de test
    """
    print("\n=== Courbe ROC pour le modèle de régression logistique ===")
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    
def tune_logistic_hyperparameters(X_train, y_train):
    """
    Ajuste les hyperparamètres d'un modèle de régression logistique à l'aide de GridSearchCV.

    :param X_train: Les données d'entraînement
    :param y_train: Les labels d'entraînement
    :return: Le meilleur modèle de régression logistique entraîné
    """
    #param_grid: Le dictionnaire des hyperparamètres à tester

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
    
    print("\n=== Meilleurs paramètres pour le modèle de régression logistique ===")
    print(grid_search.best_params_)
    return grid_search.best_estimator_
