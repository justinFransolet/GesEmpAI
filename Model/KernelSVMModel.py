from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def KernelSVMModel(X_train, y_train, X_test, y_test, method: str, c_value: float):
    """
    Entraîne un modèle SVM et analyse l'influence des caractéristiques en utilisant les coefficients des caractéristiques.

    :param X_train: Données d'entraînement
    :param y_train: Cibles d'entraînement
    :param X_test: Données de test
    :param y_test: Cibles de test
    :param method: Type de kernel (linear, rbf, poly, etc.)
    :param c_value: Paramètre de régularisation
    """
    start = time.time()

    # Initialisation et entraînement du modèle
    svm = SVC(probability=True, kernel=method, gamma="scale", C=c_value, random_state=42)
    svm.fit(X_train, y_train)

    # Prédictions
    y_pred_svm = svm.predict(X_test)
    y_proba_svm = svm.predict_proba(X_test)[:, 1]
    end = time.time()

    print(f"\nKernel SVM utilisant la méthode {method} avec une régularisation de {c_value}")
    print(f"Temps d'exécution : {end - start:.2f} secondes")

    # Évaluation
    print("\n=== Évaluation du modèle SVM ===")
    print("Matrice de Confusion :\n", confusion_matrix(y_test, y_pred_svm))
    print("\nRapport de Classification :\n", classification_report(y_test, y_pred_svm))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba_svm):.5f}")

    # Courbe ROC
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_score(y_test, y_proba_svm):.2f})", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', label="Modèle Aléatoire (AUC = 0.5)")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title(f"Courbe ROC - Kernel SVM ({method} kernel)")
    plt.legend()
    plt.grid()
    plt.show()

    # Analyse de l'influence des caractéristiques (approximée pour les kernels non-linéaires)
    if method == "linear":
        # Pour un kernel linéaire, nous avons directement accès aux coefficients
        coefficients = np.abs(svm.coef_[0])
        feature_names = X_train.columns
    else:
        # Pour les kernels non-linéaires, approximons l'importance à partir des données de support
        feature_names = X_train.columns
        support_vectors = svm.support_vectors_
        importance_scores = np.mean(np.abs(support_vectors), axis=0)
        coefficients = importance_scores

    # Créer un DataFrame pour organiser les importances
    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Importance': coefficients
    })

    # Trier les variables par importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Visualisation des importances
    print("\n=== Influence des Variables ===")
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Variable'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.title(f"Influence des Variables sur l'Attrition (SVM Kernel = {method})")
    plt.gca().invert_yaxis()  # Inverser l'ordre pour afficher les plus importantes en haut
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
