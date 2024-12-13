from imblearn.over_sampling import SMOTE

def apply_oversampling(X_train: [], y_train: []) -> ([], []):
    """
    Use SMOTE to oversample the minority classes.
    :param X_train: Features of the training set (X_train)
    :param y_train: Labels of the training set (y_train)
    :return: The oversampled training set
    """

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, y_train_res