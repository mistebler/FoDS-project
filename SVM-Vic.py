from code import *
from Feature_Selection_2 import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eval(y, X, clf, ax, fold_num, legend_entry='my legendEntry'):
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    specificity = tn / (tn + fp)
    fp_rates, tp_rates, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fp_rates, tp_rates)

    if ax is not None:
        ax.plot(fp_rates, tp_rates, label=f'Fold {fold_num} (AUC = {roc_auc:.2f})')

    return [accuracy, precision, recall, specificity, f1, roc_auc], confusion_matrix(y, y_pred)

def hypertuning(X, y, param_grid, model, cv):
    scoring = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score)}

    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=cv, verbose=1)
    clf_GS.fit(X, y)
    return clf_GS.best_estimator_

def everything(data, model, param_grid, random, kernel_name, tune_hyperparameters=False, plot=False):
    performance = pd.DataFrame(columns=['fold', 'clf', 'accuracy', 'precision', 'recall', 'specificity', 'F1', 'roc_auc'])
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cate_cols.remove('UDPYIEM')
    data_enc = pd.get_dummies(data, columns=cate_cols, drop_first=True)
    X = data_enc.drop('UDPYIEM', axis=1)
    y = data.UDPYIEM

    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 1

    if plot:
        fig_roc, axs_roc = plt.subplots(1, splits, figsize=(25, 5))
        fig_cm, axs_cm = plt.subplots(1, splits, figsize=(25, 5))

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Cast columns to float before scaling
        X_train.loc[:, num_cols] = X_train.loc[:, num_cols].astype(float)
        X_test.loc[:, num_cols] = X_test.loc[:, num_cols].astype(float)

        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_resampled.loc[:, num_cols] = scaler.fit_transform(X_train_resampled.loc[:, num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test.loc[:, num_cols])

        # Hyperparameter tuning
        if tune_hyperparameters:
            model = hypertuning(X_train_resampled, y_train_resampled, param_grid, model, cv=splits)

        model.fit(X_train_resampled, y_train_resampled)

        if plot:
            fold_metrics, cm = eval(y_test, X_test, model, axs_roc[fold - 1], fold,
                                    legend_entry=f'{kernel_name} Fold {fold}')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=axs_cm[fold - 1], cmap='Blues')
            axs_cm[fold - 1].set_title(f'Confusion Matrix (Fold {fold})')
        else:
            fold_metrics, _ = eval(y_test, X_test, model, None, fold, legend_entry=f'{kernel_name} Fold {fold + 1}')

        fold_performance = pd.DataFrame([{'fold': fold, 'clf': str(model), 'accuracy': fold_metrics[0], 'precision': fold_metrics[1],
                                          'recall': fold_metrics[2], 'specificity': fold_metrics[3], 'F1': fold_metrics[4], 'roc_auc': fold_metrics[5]}])

        performance = pd.concat([performance, fold_performance], ignore_index=True)

        fold += 1

    if plot:
        for ax in axs_roc:
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='best')
        fig_roc.suptitle(f'{kernel_name} Kernel - ROC Curves')
        fig_roc.tight_layout(rect=[0, 0, 1, 0.95])

        fig_cm.suptitle(f'{kernel_name} Kernel - Confusion Matrices')
        fig_cm.tight_layout(rect=[0, 0, 1, 0.95])


        fig_roc.savefig(f'figures/roc_curve_svm_{kernel_name}.png')
        plt.show()

        fig_cm.savefig(f'figures/confusion_matrix_svm_{kernel_name}.png')
        plt.show()

    return performance

def svm_analysis(data, random_state=42):
    splits = 5
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    performance = pd.DataFrame()

    for kernel in kernels:
        param_grid = None if kernel == 'linear' else {'kernel': [kernel], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

        if kernel == 'linear':
            model = SVC(kernel='linear', probability=True, random_state=random_state)
        else:
            model = SVC(probability=True)

        kernel_performance = everything(data, model, param_grid, random_state,
                                        kernel_name=kernel, tune_hyperparameters=(kernel != 'linear'))

        kernel_performance['kernel'] = kernel
        performance = pd.concat([performance, kernel_performance], ignore_index=True)

    # After evaluating all kernels, compute the mean performance
    mean_performance = performance.drop(columns=['clf']).groupby('kernel').mean().reset_index()
    best_kernel = mean_performance.loc[mean_performance['F1'].idxmax()]

    # Plotting the best kernel's performance
    best_kernel_name = best_kernel['kernel']
    param_grid = None if best_kernel_name == 'linear' else {'kernel': [best_kernel_name], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

    if best_kernel_name == 'linear':
        model = SVC(kernel='linear', probability=True, random_state=random_state)
    else:
        model = SVC(probability=True)

    best_kernel_performance = everything(data, model, param_grid, random_state,
                                         kernel_name=best_kernel_name, tune_hyperparameters=(best_kernel_name != 'linear'), plot=True)

    print(f"Best Kernel: {best_kernel['kernel']}")
    print(mean_performance)

    return performance, mean_performance, best_kernel_performance


# Load and clean dataset
data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
data = cleaning(data)
data = rough_filtering(data)
data.dropna(inplace=True)

# Perform SVM analysis
performance, mean_performance, best_kernel_performance = svm_analysis(data)

# Print performance tables
print("Performance Table:")
print(performance)
print("\nMean Performance Table:")
print(mean_performance)

# Save performance table to CSV
best_kernel_performance.to_csv('performance_SVM.csv', index=False)
