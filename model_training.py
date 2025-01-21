from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import shap
from visualizations import save_confusion_matrix, plot_shap, plot_roc_curve, plot_pr_curve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


def train_models(X, y, columns, models, acc_lists, cv_list, odor):
    """
    Train multiple models using specified cross-validation strategies, evaluate performance,
    and generate visualizations for metrics and feature importance.

    Args:
        X (ndarray): Input feature matrix.
        y (ndarray): Target labels.
        columns (list): Feature names.
        models (list): List of machine learning models to train.
        acc_lists (dict): Dictionary to store accuracy scores for each CV setting.
        cv_list (list): List of cross-validation strategies.
    """
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    for model in models:
        for cv_count, label in cv_list:
            kf = StratifiedKFold(n_splits=cv_count, shuffle=True, random_state=29)
            accuracy_scores, classification_reports, confusion_matrices, tprs, aucs, precisions_list, average_precision_list = [], [], [], [], [], [], []
            base_recall = np.linspace(0, 1, 100)
            mean_fpr = np.linspace(0, 1, 100)
            class_names = ["Ammonia", "Rancid"]
            shap_values_list = {0: [], 1: []}

            for train_index, test_index in kf.split(X, y):
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]
                n_samples = test_X.shape[0]

                # Train model
                model.fit(train_X, train_y)
                y_pred = model.predict(test_X)
                y_proba = model.predict_proba(test_X)[:, 1]

                # Compute metrics
                accuracy = balanced_accuracy_score(test_y, y_pred)
                accuracy_scores.append(accuracy)

                fpr, tpr, _ = roc_curve(test_y, y_proba)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

                precision, recall, _ = precision_recall_curve(test_y, y_proba)
                interp_precision = interp1d(recall, precision, kind='linear', bounds_error=False,
                                            fill_value=(precision[0], precision[-1]))
                precisions_list.append(interp_precision(base_recall))

                average_precision = average_precision_score(test_y, y_proba)
                average_precision_list.append(average_precision)

                acc_lists[f'accuracy_list_{cv_count}'].append(accuracy)

                # Compute confusion matrix
                cm = confusion_matrix(test_y, y_pred)
                confusion_matrices.append(cm)

                # Compute SHAP values
                explainer = shap.KernelExplainer(model.predict_proba, train_X)
                shap_values = explainer.shap_values(test_X)
                for i in range(len(shap_values)):
                    shap_values_list[i].append(shap_values[i])

            # Generate and save plots
            plot_shap(shap_values_list, X, columns, model, cv_count, class_names, odor)
            plot_roc_curve(ax1, tprs, mean_fpr, label, aucs, cv_count, model, odor)
            plot_pr_curve(precisions_list, base_recall, average_precision_list, cv_count, model, odor)

            # Save confusion matrices for each fold
            for fold, cm in enumerate(confusion_matrices, 1):
                save_confusion_matrix(fold, model, cm, cv_count, odor)

    # Save the overall ROC curve
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', alpha=0.8, label='Chance')
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], title="Model Performance Comparison Across CV Settings")
    ax1.legend(loc="lower right")
    output_dir = f'./result/{odor}/roc'
    os.makedirs(output_dir, exist_ok=True)
    fig1.savefig(f'{output_dir}/roc_curve.png')
    plt.close(fig1)