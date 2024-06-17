import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
import matplotlib.pyplot as plt
from Feature_Selection_2 import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


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


# Evaluation function
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

    ax.plot(fp_rates, tp_rates, label=f'Classifier fold {legend_entry}')

    # Return confusion matrix
    return [accuracy, precision, recall, specificity, f1, roc_auc], confusion_matrix(y, y_pred)


# Feature selection function
def selection(X_train, y_train, how, n, what):
    UVFS_Selector = SelectKBest(score_func=how, k=n)
    X_selected = UVFS_Selector.fit_transform(X_train, y_train)
    if what == 'scores':
        return UVFS_Selector.scores_
    elif what == 'features':
        return UVFS_Selector.get_feature_names_out()
    elif what == 'mutual_info':
        mutual_info = mutual_info_classif(X_train, y_train)
        mutual_info = pd.Series(mutual_info, index=X_train.columns).sort_values(ascending=False)
        return mutual_info


# Plotting function
def picture(x, y, features, ax):
    ax.bar(x, y)
    ax.set_xticks(ticks=x, labels=features, rotation=90)
    ax.set_title(input('Title: '))


# Hyperparameter tuning function
def hypertuning(X, y, parameters, model, cv):
    param_grid = parameters
    scoring = {'precision': 'precision', 'f1': 'f1', 'recall': 'recall'}
    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='f1', cv=cv)
    clf_GS.fit(X, y)
    return clf_GS.best_estimator_


# Main function for model training and evaluation
def everything(data, model, param, random, tune_hyperparameters=True):
    performance = pd.DataFrame(
        columns=['fold', 'clf', 'accuracy', 'precision', 'recall', 'specificity', 'F1', 'roc_auc'])
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cate_cols.remove('UDPYIEM')

    data_enc = pd.get_dummies(data, columns=cate_cols, drop_first=True)
    X = data_enc.drop('UDPYIEM', axis=1)
    y = data['UDPYIEM']

    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 0

    fig, axs = plt.subplots(1, splits, figsize=(15, 5))
    model_names = []

    X.loc[:, num_cols] = X.loc[:, num_cols].astype(float)

    confusion_matrices = []  # To store confusion matrices

    for train_index, test_index in cv.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("\nClass Distribution Before Undersampling:")
        print(pd.Series(y_train).value_counts())

        #smote = SMOTE(random_state=42)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        #X_train = X_train.astype(float)
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

        # Feature selection
        selected_features = selection(X_train_resampled, y_train_resampled, mutual_info_classif, 45, "features")
        X_train_resampled = X_train_resampled[selected_features]
        X_test = X_test[selected_features]

        print(selected_features)

        num_cols = X_train_resampled.select_dtypes(include=['int64', 'float64']).columns.tolist()

        print("\nClass Distribution After Undersampling:")
        print(pd.Series(y_train_resampled).value_counts())

        scaler = StandardScaler()

        X_train_resampled.loc[:, num_cols] = scaler.fit_transform(X_train_resampled.loc[:, num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test.loc[:, num_cols])



        best_model = hypertuning(X_train_resampled, y_train_resampled, param, model, cv=splits)

        metrics, cm = eval(y_test, X_test, best_model, axs[fold], fold_num=fold, legend_entry=f'fold {fold + 1}')
        performance.loc[fold] = [fold, best_model] + list(metrics)
        confusion_matrices.append(cm)  # Add confusion matrix to list
        fold += 1

    for i, ax in enumerate(axs):
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC curve for fold {i + 1}')
        ax.plot([0, 1], [0, 1], color='r', ls='--', label='random classifier')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/roc_curve_knearest.png')
    plt.show()

    # Plot all confusion matrices together
    fig_cm, axs_cm = plt.subplots(1, splits, figsize=(15, 5))
    for i, cm in enumerate(confusion_matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs_cm[i])
        axs_cm[i].set_title(f'Confusion Matrix for Fold {i + 1}')
        axs_cm[i].set_xlabel('Predicted Label')
        axs_cm[i].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrices_knearest.png')
    plt.show()

    return performance


# Specific function to perform KNN analysis
def knn_analysis(data, random_state=42):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 21], 'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan', 'minkowski']}
    knn = KNeighborsClassifier()
    performance = everything(data=data, model=knn, param=param_grid, random=random_state)
    return performance
def selection(X_train, y_train, how, n, what):
    UVFS_Selector = SelectKBest(score_func=how, k=n)
    X_selected = UVFS_Selector.fit_transform(X_train, y_train)
    if what == 'scores':
        return UVFS_Selector.scores_
    elif what == 'features':
        return UVFS_Selector.get_feature_names_out()
    elif what == 'mutual_info':
        mutual_info = mutual_info_classif(X_train, y_train)
        mutual_info = pd.Series(mutual_info, index=X_train.columns).sort_values(ascending=False)
        return mutual_info



# Run KNN analysis
performance = knn_analysis(data)
print(performance)
performance.to_csv('performance_KNear.csv', index=False)
