import os
import matplotlib.pyplot as plt
import numpy as np
import shap
import itertools


def save_confusion_matrix(fold, model, cm, cv_count, odor):
    """
    Save the confusion matrix plot for a specific fold and model.

    Args:
        fold (int): Fold index.
        model: Machine learning model instance.
        cm (ndarray): Confusion matrix.
        cv_count (int): Number of cross-validation folds.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model.__class__.__name__}({cv_count}-Fold) - Fold {fold}')
    plt.colorbar()

    # Define class names
    classes = ["Ammonia", "Rancid"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate cells with values
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save plot to file
    output_dir = f'./result/{odor}/cm/{model.__class__.__name__}'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/({cv_count}-Fold)-Fold{fold}.png')
    plt.close(fig)


def plot_shap(shap_values_list, X, columns, model, cv_count, class_names, odor):
    """
    Generate and save SHAP summary plots for feature importance.

    Args:
        shap_values_list (dict): Dictionary of SHAP values for each class.
        X (ndarray): Input feature matrix.
        columns (list): Feature names.
        model: Machine learning model instance.
        cv_count (int): Number of cross-validation folds.
        class_names (list): List of class names.
    """
    mean_shap_values = {}
    for i in shap_values_list:
        # Aggregate SHAP values across all folds
        all_shap_values = np.vstack(shap_values_list[i])
        mean_shap_values[i] = np.mean(all_shap_values, axis=0)
    for i in mean_shap_values:
        if mean_shap_values[i].ndim == 1:
            mean_shap_values[i] = mean_shap_values[i][np.newaxis, :]
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        shap.summary_plot(mean_shap_values[i], X, feature_names=columns, plot_type="bar", show=False)
        output_dir = f'./result/{odor}/shap/{model.__class__.__name__}'
        os.makedirs(output_dir, exist_ok=True)
        fig2.savefig(f'{output_dir}/{cv_count}fold_bar_{class_names[i]}.png')
        plt.close(fig2)


def plot_roc_curve(ax1, tprs, mean_fpr, label, aucs, cv_count, model, odor):
    """
    Generate and save ROC curve plots.

    Args:
        ax1: Axes for the ROC plot.
        tprs (list): List of true positive rates across folds.
        mean_fpr (ndarray): Mean false positive rates.
        label (str): Label for the plot.
        aucs (list): List of AUC scores across folds.
        cv_count (int): Number of cross-validation folds.
        model: Machine learning model instance.
    """
    fig_model, ax_model = plt.subplots()
    model_mean_tpr = np.mean(tprs, axis=0)
    model_mean_tpr[-1] = 1.0
    model_mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # Plot the mean ROC curve
    ax1.plot(mean_fpr, model_mean_tpr,
             label=f'{model.__class__.__name__} {label} (AUC = {model_mean_auc:.2f} ± {std_auc:.2f})', lw=2,
             alpha=0.8)
    ax_model.plot(mean_fpr, model_mean_tpr,
                  label=f'{cv_count}-Fold (AUC = {model_mean_auc:.2f} ± {std_auc:.2f})')
    ax_model.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', alpha=0.8, label='Chance')
    ax_model.set_title(f'{model.__class__.__name__} Average ROC Curve')
    ax_model.set_xlabel('False Positive Rate')
    ax_model.set_ylabel('True Positive Rate')
    ax_model.legend(loc='lower right')

    # Save plot to file
    output_dir = f'./result/{odor}/roc/{model.__class__.__name__}'
    os.makedirs(output_dir, exist_ok=True)
    fig_model.savefig(f'{output_dir}/{cv_count}-fold_roc_curve.png')
    plt.close(fig_model)


def plot_pr_curve(precisions_list, base_recall, average_precision_list, cv_count, model, odor):
    """
    Generate and save precision-recall (PR) curve plots.

    Args:
        precisions_list (list): List of precision values across folds.
        base_recall (ndarray): Recall values for interpolation.
        average_precision_list (list): List of average precision scores across folds.
        cv_count (int): Number of cross-validation folds.
        model: Machine learning model instance.
    """
    fig_model_pr, ax_model_pr = plt.subplots()
    mean_precision = np.mean(precisions_list, axis=0)
    ax_model_pr.plot(base_recall, mean_precision, label=f'{cv_count}-Fold (AP = {np.mean(average_precision_list):.2f})')
    ax_model_pr.set_title(f'{model.__class__.__name__} Average Precision-Recall Curve')
    ax_model_pr.set_xlabel('Recall')
    ax_model_pr.set_ylabel('Precision')
    ax_model_pr.legend(loc='lower left')

    # Save plot to file
    output_dir = f'./result/{odor}/pr/{model.__class__.__name__}'
    os.makedirs(output_dir, exist_ok=True)
    fig_model_pr.savefig(f'{output_dir}/{cv_count}-fold_pr_curve.png')
    plt.close(fig_model_pr)