from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def DecisionTreeModel(X_train, y_train, X_test, y_test):
    """
    Fonction pour entraîner, optimiser et tester un modèle d'arbre de décision.

    :param X_train: Données d'entraînement
    :param y_train: Labels d'entraînement
    :param X_test: Données de test
    :param y_test: Labels de test
    :return: None
    """

    # Initialisation du modèle
    tree_model = DecisionTreeClassifier(max_depth=None, random_state=42)

    # Entraînement et évaluation
    tree_model.fit(X_train, y_train)
    y_pred_initial = tree_model.predict(X_test)

    print("\n=== Résultats de base ===")
    print("Matrice de Confusion :\n", confusion_matrix(y_test, y_pred_initial))
    print("Rapport de Classification :\n", classification_report(y_test, y_pred_initial))

    if hasattr(tree_model, "predict_proba"):
        y_proba_initial = tree_model.predict_proba(X_test)[:, 1]
        roc_auc_initial = roc_auc_score(y_test, y_proba_initial)
        print("AUC-ROC initial :", roc_auc_initial)

    # Optimisation avec GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 8, 10, 20, None],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\n=== Résultats de GridSearchCV ===")
    print("Meilleurs paramètres :", grid_search.best_params_)
    print("Meilleure précision (cross-validation) :", grid_search.best_score_)

    # Optimisation avec RandomizedSearchCV
    param_distributions = {
        'max_depth': [3, 5, 8, 10, 20, None],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'criterion': ['gini', 'entropy']
    }
    random_search = RandomizedSearchCV(estimator=tree_model, param_distributions=param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    print("\n=== Résultats de RandomizedSearchCV ===")
    print("Meilleurs paramètres :", random_search.best_params_)
    print("Meilleure précision (cross-validation) :", random_search.best_score_)

    # Validation croisée avec le modèle optimisé
    best_tree = DecisionTreeClassifier(**random_search.best_params_, random_state=42)
    cv_scores = cross_val_score(best_tree, X_train, y_train, cv=5, scoring='accuracy')

    print("\n=== Validation croisée avec le modèle optimisé ===")
    print("Scores de validation croisée :", cv_scores)
    print("Score moyen :", cv_scores.mean())

    # Évaluation finale sur le jeu de test
    best_tree.fit(X_train, y_train)
    y_pred_final = best_tree.predict(X_test)

    print("\n=== Résultats finaux ===")
    print("Matrice de Confusion :\n", confusion_matrix(y_test, y_pred_final))
    print("Rapport de Classification :\n", classification_report(y_test, y_pred_final))

    if hasattr(best_tree, "predict_proba"):
        y_proba_final = best_tree.predict_proba(X_test)[:, 1]
        roc_auc_final = roc_auc_score(y_test, y_proba_final)
        print("AUC-ROC final :", roc_auc_final)

        # Tracer la courbe ROC
        fpr_final, tpr_final, thresholds_final = roc_curve(y_test, y_proba_final)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_final, tpr_final, label=f"Decision Tree Optimisé (AUC = {roc_auc_final:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Aléatoire (AUC = 0.5)")
        plt.xlabel("Taux de Faux Positifs")
        plt.ylabel("Taux de Vrais Positifs")
        plt.title("Courbe ROC - Modèle Optimisé")
        plt.legend()
        plt.show()
    else:
        print("Le modèle ne fournit pas de probabilités pour les classes.")