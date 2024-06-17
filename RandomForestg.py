import sklearn
#from Feature_Selection_2 import  cleaning, rough_filtering ,eval, everything
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  precision_recall_curve
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler

def cleaning(data):
    data['UDPYIEM'] = pd.Categorical(data['UDPYIEM'])
    none_991 = ['IRCIGAGE', 'IRCDUAGE', 'IRCGRAGE', 'IRSMKLSSTRY', 'IRALCAGE', 'IRMJAGE', 'IRTOBAGE','IRMJALLGAGE','ILLICITAGE']
    data['BLNTAGE'] = data['BLNTAGE'].where(data['BLNTAGE'] < 70, None).astype('Int64')
    binary = ['CIGFLAG','CGRFLAG','PIPFLAG', 'SMKLSSFLAG', 'TOBFLAG', 'ALCFLAG','MRJFLAG','DCIGMON','FUCIG18','FUCIG21', 'FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','DEPNDALC','DEPNDMRJ', 'ABUSEALC','ABUSEMRJ','ABODALC','ABODMRJ','CDUFLAG','CDCGMO','BNGDRKMON','HVYDRKMON','FUCD218','FUCD221','DNICNSP','CIGDLYMO','CIG100LF','PIPE30DY','BLNTNOMJ','BLNTEVER']
    ordinal = ['CIG30TPE','ALCYDAYS','MRJYDAYS','CIGMDAYS','CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','BNGDRMDAYS','CIGAVGD', 'AGEALC','AGETOB','AGEMRJ']
    for i in none_991:
        data[i] = data[i].replace({991:None, 993:None, 999:None})
    for i in ordinal:
        if i == 'CIGAVGD':
            data[i] = pd.Categorical(data[i], ordered=True)
        else:
            data[i] = pd.Categorical(data[i].astype('Int64'), ordered=True)
    for i in binary:
        data[i] = pd.Categorical(data[i].astype('Int64'))

    convert = ['ALCUS30D','IRCIGAGE', 'IRCDUAGE', 'IRCGRAGE', 'IRSMKLSSTRY', 'IRALCAGE','IRMJAGE', 'BLNTAGE', 'BLNT30DY','IRTOBAGE','IRMJALLGAGE']
    for i in convert:
        data[i] = data[i].astype('Int64')
    data.drop(['IRCOCAGE','IRCRKAGE','IRHERAGE','IRHALLUCAGE','IRLSDAGE','IRPCPAGE','IRECSTMOAGE','IRINHALAGE','IRMETHAMAGE','IRPNRNMAGE','IRTRQNMAGE','IRSTMNMAGE','IRSEDNMAGE'], axis=1, inplace=True)
    data.drop(['BLNTNOMJ'], axis=1, inplace=True)
    return data
def rough_filtering(df):
    df.drop(['IRSMKLSSTRY','IRCGRAGE','IRCIGAGE','BLNTAGE','IRMJAGE','IRCDUAGE'],axis=1,inplace=True)
    #Alter rausnehmen --> befindet sich in neuen Features als ordinal data
    df.drop(['IRTOBAGE', 'IRMJALLGAGE', 'IRALCAGE','ILLICITAGE'], axis=1, inplace=True)
    df.dropna(inplace=True) #war vorher unten falls ihr es nicht selber implementiert habt dann hatte es noch missing values in eurem Datenset (sorry)
    return df
def everything(data, model, param, random, tune_hyperparameters=True):

    #initiallizing
    performance = pd.DataFrame(
        columns=['fold', 'clf', 'accuracy', 'precision', 'recall', 'specificity', 'F1', 'roc_auc'])
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    cate_cols.remove('UDPYIEM')

    X = data.drop('UDPYIEM', axis=1)
    y = data['UDPYIEM']

    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 0

    fig, axs = plt.subplots(1, splits, figsize=(15, 5))

    X.loc[:, num_cols] = X.loc[:, num_cols].astype(float)

    confusion_matrices = []

    feature_importance_all = pd.DataFrame({"Feature": X.columns})

    #Stratified K fold
    for train_index, test_index in cv.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

        #Feature selection
        selected_features = selection(X_train_resampled, y_train_resampled, mutual_info_classif, 45, "features")
        selected_features = Random_forest_feature_selection(X_train_resampled[selected_features], y_train_resampled, 0.01)
        X_train_resampled = X_train_resampled[selected_features]
        X_test = X_test[selected_features]

        num_cols = X_train_resampled.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cate_cols = X_train_resampled.select_dtypes(include=['object', 'category']).columns.tolist()
        scaler = StandardScaler()

        X_train_resampled.loc[:, num_cols] = scaler.fit_transform(X_train_resampled.loc[:, num_cols])
        X_test.loc[:, num_cols] = scaler.transform(X_test.loc[:, num_cols])

        #Hyperparameter
        best_model = hypertuning(X_train_resampled, y_train_resampled, param, model, cv=splits)

        feature_importance_this_fold = RF_feature_importance(best_model, X_train_resampled, fold)
        feature_importance_all = pd.concat([feature_importance_all, feature_importance_this_fold], axis=1)

        metrics, cm = eval(y_test, X_test, best_model, axs[fold], fold_num=fold, legend_entry=f'fold {fold + 1}')
        performance.loc[fold] = [fold, best_model] + list(metrics)
        confusion_matrices.append(cm)
        fold += 1

    feature_importance_all["average"] = feature_importance_all[["Importance fold 0", "Importance fold 1", "Importance fold 2", "Importance fold 3","Importance fold 4"]].mean(axis=1)
    feature_importance_all["Std"] = feature_importance_all[["Importance fold 0", "Importance fold 1", "Importance fold 2", "Importance fold 3","Importance fold 4"]].std(axis=1)
    feature_importance_all.sort_values(by=['average'], ascending=False, inplace=True)
    print("Top 5 feature importance")
    print(feature_importance_all[["Feature", "average", "Std"]].head(5))

    #plotting

    # ROC
    for i, ax in enumerate(axs):
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC curve for fold {i + 1}')
        ax.plot([0, 1], [0, 1], color='r', ls='--', label='random classifier')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/roc_curve_RF.png')
    #feature importance

    plt.figure()
    sns.barplot(x='Feature', y='average', data=feature_importance_all)
    plt.errorbar(x=feature_importance_all['Feature'], y=feature_importance_all['average'], yerr=feature_importance_all['Std'],fmt = "none", c='black')
    plt.title('Feature Importance RF')
    plt.ylabel("Average Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/Feature_importance_RF.png')

    #Confusion Matrices
    fig_cm, axs_cm = plt.subplots(1, splits, figsize=(15, 5))
    for i, cm in enumerate(confusion_matrices):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs_cm[i])
        axs_cm[i].set_title(f'Confusion Matrix for Fold {i + 1}')
        axs_cm[i].set_xlabel('Predicted Label')
        axs_cm[i].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('figures/confusion_matrices_RF.png')
    plt.show()

    return performance

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

def hypertuning(X,y,parameters,model, cv):
    splits = cv
    cv = StratifiedKFold(n_splits=splits, shuffle=True)
    param_grid = parameters
    scoring = {'precision': make_scorer(precision_score),'f1': make_scorer(f1_score),'recall': make_scorer(recall_score)}
    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit = "f1")
    clf_GS.fit(X,y)
    return clf_GS.best_estimator_

#Feature selection
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
def Random_forest_feature_selection(X_train,y_train,threshold):
    feature_model = RandomForestClassifier(random_state=42)
    feature_model.fit(X_train, y_train)
    feature_importance = pd.DataFrame({"Feature": X_train.columns,
                                       "Importance": feature_model.feature_importances_ })
    feature_importance_sorted = feature_importance.sort_values(by="Importance", ascending=False)
    important_features = []
    for i, row in feature_importance_sorted.iterrows():
        if row["Importance"] >= threshold:
            important_features.append(row["Feature"])
    return important_features

    #reduces features from 58 to 37
def RF_feature_importance(model, X, fold):
    feature_importances = model.feature_importances_

    features = pd.DataFrame()
    features['Feature'] = X.columns
    features[f'Importance fold {fold}'] = feature_importances

    return features[f'Importance fold {fold}']

#importing data
data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)

#preprocessing
data = cleaning(data)
data = rough_filtering(data)

#Model
model = RandomForestClassifier(random_state=42)

#Hyperparameter
param = {"n_estimators": [50,100,200], "max_depth": [None, 2, 5, 10], "min_samples_split": [2, 5, 10]}
performance = everything(data, model, param, 42)

#performance display
pd.set_option("display.max_colwidth",None)
print("performance")
print(performance)
print("Average performance")
print(f"hyperparameter:{performance['clf']}")
print(f"accuracy:{performance['accuracy'].mean()} +- {performance['accuracy'].std()}")
print(f"precision:{performance['precision'].mean()} +- {performance['precision'].std()}")
print(f"recall:{performance['recall'].mean()} +- {performance['recall'].std()}")
print(f"specificity:{performance['specificity'].mean()} +- {performance['specificity'].std()}")
print(f"F1:{performance['F1'].mean()} +- {performance['F1'].std()}")
print(f"roc_auc:{performance['roc_auc'].mean()} +- {performance['roc_auc'].std()}")