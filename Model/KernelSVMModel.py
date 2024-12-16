from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def KernelSVMModel(x_train: [], y_train: [], x_test: [], y_test: [],method: str, c_value: float) -> None:
    """
    This function is used to train and test the Kernel SVM model

    :param x_train: Data to train the model
    :param y_train: Target to train the model
    :param x_test: Data to test the model
    :param y_test: Target to test the model
    :param method: Method used to train the model
    :param c_value: Regularization parameter
    """
    # Initialisation et entraînement du modèle
    svm = SVC(probability=True, kernel=method,gamma="scale", C=c_value)
    svm.fit(x_train, y_train)

    # Prédictions
    y_pred_svm = svm.predict(x_test)

    print(f"\nKernel SVM utilisant la méthode {method} avec un régulation de {c_value}\n")
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
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, svm.predict_proba(x_test)[:, 1])
    plt.plot(fpr_svm, tpr_svm, label="SVM")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.show()