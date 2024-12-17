import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def tune_xgboost_hyperparameters(X_train, y_train):
    """
    Ajuste les hyperparamètres pour XGBoost avec GridSearchCV.

    :param X_train: Données d'entraînement
    :param y_train: Labels d'entraînement
    :return: Modèle optimisé
    """
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("\n=== Meilleurs paramètres ===")
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def train_xgboost_model(X_train, y_train, params=None):
    """
    Entraîne un modèle XGBoost sur les données d'entraînement.

    :param X_train: Les données d'entraînement
    :param y_train: Les labels d'entraînement
    :param params: Dictionnaire des hyperparamètres (optionnel)
    :return: Le modèle XGBoost entraîné
    """
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'max_depth': 4,             # Augmenté légèrement pour capturer plus de complexité
            'learning_rate': 0.05,      # Réduit pour un apprentissage plus fin
            'n_estimators': 300,        # Augmenté pour plus d'arbres
            'subsample': 0.8,           # Échantillonnage partiel pour éviter le sur-apprentissage
            'colsample_bytree': 0.8     # Sélection partielle des caractéristiques
        }

    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def predict_xgboost_model(model, X_test):
    """
    Effectue des prédictions avec un modèle XGBoost.

    :param model: Le modèle XGBoost entraîné
    :param X_test: Les données de test
    :return: Les prédictions et les probabilités pour la classe positive
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def evaluate_xgboost_model(y_test, y_pred, y_proba):
    """
    Évalue les performances d'un modèle XGBoost.

    :param y_test: Les labels réels de test
    :param y_pred: Les prédictions du modèle
    :param y_proba: Les probabilités pour la classe positive
    """
    print("\n=== Évaluation du modèle XGBoost ===")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.5f}")
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("Courbe ROC pour le modèle XGBoost")
    plt.show()




