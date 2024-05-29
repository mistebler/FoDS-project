from code import *
from Feature_Selection_2 import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay


#integrate other kernel for non-linear!!!!
#should i do feature selection in svm_analysis function after hyperparameter tunin?
#everything function und svm_analysis function zusammenführen
def eval(y, X, clf, ax_roc, ax_cm, legend_entry='my legendEntry'):
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

    # Plot ROC curve
    ax_roc.plot(fp_rates, tp_rates, label=f'{legend_entry} (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='r', ls='--', label='Random Classifier')
    ax_roc.legend(loc='best')

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap='Blues')
    ax_cm.set_title(f'Confusion Matrix ({legend_entry})')

    return [accuracy, precision, recall, specificity, f1, roc_auc]

def hypertuning(X,y,param_grid,model,cv):
    scoring = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score)}

    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=cv, verbose=1)
    #clf_GS = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring, refit='accuracy', cv=cv, n_iter=5, n_jobs=-1, verbose=1)
    clf_GS.fit(X, y)
    # nachher hyptertuning().etc nehmen
    #gibt den besten estimator an --> z.B. um y_prediction zu bekommen hyptertuning(...).predict(X) angeben
    return clf_GS.best_estimator_


def everything(data, model, param_grid, random, axs_roc, axs_cm, kernel_name, tune_hyperparameters=False):
    performance = pd.DataFrame(columns=['fold', 'clf', 'accuracy', 'precision', 'recall', 'specificity', 'F1', 'roc_auc'])
    num_cols = data.select_dtypes(include=['Int64', 'float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cate_cols.remove('UDPYIEM')
    data_enc = pd.get_dummies(data, columns=cate_cols, drop_first=True)
    X = data_enc.drop('UDPYIEM', axis=1)
    y = data.UDPYIEM

    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 0

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

        ax_roc_fold = axs_roc[fold] if isinstance(axs_roc, np.ndarray) else axs_roc
        ax_cm_fold = axs_cm[fold] if isinstance(axs_cm, np.ndarray) else axs_cm

        # Added kernel name to legend entry
        fold_metrics = eval(y_test, X_test, model, ax_roc_fold, ax_cm_fold, legend_entry=f'{kernel_name} Fold {fold + 1}')

        fold_performance = pd.DataFrame([{'fold': fold, 'clf': str(model), 'accuracy': fold_metrics[0], 'precision': fold_metrics[1],
                                          'recall': fold_metrics[2], 'specificity': fold_metrics[3], 'F1': fold_metrics[4], 'roc_auc': fold_metrics[5]}])

        performance = pd.concat([performance, fold_performance], ignore_index=True)

        fold += 1
    return performance
def svm_analysis(data, random_state=42):
    splits = 5
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']  # List of kernels
    performance = pd.DataFrame()

    fig_roc, axs_roc = plt.subplots(1, splits, figsize=(20, 5))
    fig_cm, axs_cm = plt.subplots(1, splits, figsize=(20, 5))

    best_kernel_performance = None  # Store performance metrics for the best kernel

    for row, kernel in enumerate(kernels):
        param_grid = None if kernel == 'linear' else {'kernel': [kernel], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

        if kernel == 'linear':
            model = SVC(kernel='linear', probability=True, random_state=random_state)
        else:
            model = SVC(probability=True)

        kernel_performance = everything(data, model, param_grid, random_state, axs_roc[row], axs_cm[row],
                                        kernel_name=kernel, tune_hyperparameters=(kernel != 'linear'))

        kernel_performance['kernel'] = kernel
        performance = pd.concat([performance, kernel_performance], ignore_index=True)

        # If the current kernel is the best one, store its performance metrics
        if kernel == mean_performance['kernel'].iloc[0]:
            best_kernel_performance = kernel_performance

    for i, ax in enumerate(axs_roc):
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{best_kernel_performance["kernel"].iloc[0]} Kernel - Fold {i + 1}')
        fp_rates = best_kernel_performance.loc[i, 'fp_rates']
        tp_rates = best_kernel_performance.loc[i, 'tp_rates']
        ax.plot(fp_rates, tp_rates, label=f'{best_kernel_performance["kernel"].iloc[0]} (AUC = {best_kernel_performance["roc_auc"].iloc[0]:.2f})')
        ax.plot([0, 1], [0, 1], color='r', ls='--', label='Random Classifier')
        ax.legend(loc='best')

    for i, ax in enumerate(axs_cm):
        ax.set_title(f'Confusion Matrix ({best_kernel_performance["kernel"].iloc[0]} Kernel - Fold {i + 1})')
        cm = best_kernel_performance.loc[i, 'confusion_matrix']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues')

    plt.tight_layout()
    plt.savefig('figures/roc_curve_best_kernel.png')
    plt.show()

    plt.tight_layout()
    plt.savefig('figures/confusion_matrices_best_kernel.png')
    plt.show()

    # Determine the best kernel based on mean performance metrics across all folds
    mean_performance = performance.drop(columns=['clf']).groupby('kernel').mean().reset_index()
    best_kernel = mean_performance.loc[mean_performance['accuracy'].idxmax()]

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