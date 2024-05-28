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
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax_cm)
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
    undersampler = RandomUnderSampler(random_state=42)
    X, y = undersampler.fit_resample(X, y)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]



        # Cast columns to float before scaling
        X_train.loc[:, num_cols] = X_train.loc[:, num_cols].astype(float)
        X_test.loc[:, num_cols] = X_test.loc[:, num_cols].astype(float)

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # Hyperparameter tuning
        if tune_hyperparameters:
            model = hypertuning(X_train, y_train, param_grid, model, cv=splits)

        model.fit(X_train, y_train)

        # Added kernel name to legend entry
        fold_metrics = eval(y_test, X_test, model, axs_roc[fold], axs_cm[fold], legend_entry=f'{kernel_name} Fold {fold + 1}')

        fold_performance = pd.DataFrame([{'fold': fold, 'clf': str(model), 'accuracy': fold_metrics[0], 'precision': fold_metrics[1],
                                          'recall': fold_metrics[2], 'specificity': fold_metrics[3], 'F1': fold_metrics[4], 'roc_auc': fold_metrics[5]}])

        performance = pd.concat([performance, fold_performance], ignore_index=True)

        fold += 1
    return performance
def svm_analysis(data, random_state=42):
    splits = 5
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']  # List of kernels
    performance = pd.DataFrame()

    fig_roc, axs_roc = plt.subplots(len(kernels), splits, figsize=(20, 15))
    fig_cm, axs_cm = plt.subplots(len(kernels), splits, figsize=(20, 15))

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

    for i, row in enumerate(axs_roc):
        for j, ax in enumerate(row):
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{kernels[i]} Kernel - Fold {j + 1}')

    plt.tight_layout()
    plt.savefig('figures/roc_curve_svm.png')
    plt.show()

    fig_cm.tight_layout()
    fig_cm.savefig('figures/confusion_matrices_svm.png')
    plt.show()

    return performance
#Do Gateway Drugs (Marjuhaha, Alcohol, Tobaco) correlate with future substance abuse disorder
# Load and clean your dataset

data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
data = cleaning(data)
data = rough_filtering(data)
data.dropna(inplace=True)

# Identify categorical, ordinal, and binary features
ordinal = []
binary = []
categorical = data.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical:
    if data[col].cat.ordered:
        ordinal.append(col)
    else:
        binary.append(col)
binary.remove('UDPYIEM')

performance = svm_analysis(data)
print(performance)