from sklearn.metrics import roc_curve, auc, confusion_matrix

def auc_score(y, y_score):
    """Metric AUROC, y, y_score needn't to be sorted in increasing or decreasing 
    order.

    Args:
        y (np.ndarray): label 
        y_score (np.ndarray): positive score
    """
    result = sorted(list(zip(y, y_score)), key=lambda x: x[0])
    # print(result)
    y = [i[0] for i in result]
    y_score = [i[1][0] for i in result]
    fpr, tpr, threshold = roc_curve(y, y_score)
    print("AUROC: {}".format(auc(fpr, tpr)))


def sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print("Sensitivity: {}, Specificity: {}".format(sensitivity, specificity))