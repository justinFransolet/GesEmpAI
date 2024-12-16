from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def KernelSVMModel(X_train: [], y_train: [], X_test: [], y_test: []) -> None:
    # Initialisation et entraînement du modèle
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    # Prédictions
    y_pred_svm = svm.predict(X_test)

    # Matrice de confusion
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    print("Matrice de Confusion (SVM) :\n", conf_matrix_svm)

    # Rapport de classification
    class_report_svm = classification_report(y_test, y_pred_svm)
    print("Rapport de Classification (SVM) :\n", class_report_svm)

    # AUC-ROC
    roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
    print("AUC-ROC (SVM) :\n", roc_auc_svm)

    # Courbe ROC
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, svm.predict_proba(X_test)[:,1])
    plt.plot(fpr_svm, tpr_svm, label="SVM")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.show()