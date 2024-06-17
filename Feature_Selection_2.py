import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import chi2, f_classif, SelectKBest, mutual_info_classif, VarianceThreshold, SelectFromModel
import scipy.stats as sts
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  precision_recall_curve, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes
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

#print(data.select_dtypes(include='float64').columns.tolist())

"""
anschauen = []

for i in data.columns.tolist():
    if i in data.select_dtypes(include='Int64').columns:
        if data.loc[data[i]>90,i].shape[0]!=0:
            anschauen.append(i)
    elif i in data.select_dtypes(include='float64').columns:
        if data.loc[data[i]>90.0,i].shape[0]!=0:
            anschauen.append(i)
    if i in ordinal or i in binary:
        if data.loc[data[i].astype('Float64')>90.0,i].shape[0]!=0:
            anschauen.append(i)
print(anschauen)
"""
#for i in data.select_dtypes(include='category').columns.tolist():
    #print(data[i].value_counts())
#for col in data.columns:
        #if data[col].dtype == "category":
            #print(f"Column {col} ordered? {data[col].cat.ordered}")

#Feature Continuous --> Anova
#Feature Categorical --> Chi-Square
data = cleaning(data)
#print(data.isna().sum().sort_values(ascending=False)/data.shape[0])

#Grobe Feature selection (meisten Nan values)
plt.figure(figsize=(14,6))

plt.bar(data.isna().sum().sort_values(ascending=False).index, data.isna().sum().sort_values(ascending=False)/data.shape[0])
plt.xticks(data.isna().sum().sort_values(ascending=False).index, rotation=90)
plt.ylabel('% missing values')
plt.xlabel('Features')
plt.title('Amount of missing values')
plt.tight_layout()
plt.savefig('figures/missing_values.png')
#print(IRMJALLGAGE.isna().sum()/data.shape[0])
pd.set_option("display.max_columns", data.shape[1])
def rough_filtering(df):
    df.drop(['IRSMKLSSTRY','IRCGRAGE','IRCIGAGE','BLNTAGE','IRMJAGE','IRCDUAGE'],axis=1,inplace=True)
    #Alter rausnehmen --> befindet sich in neuen Features als ordinal data
    df.drop(['IRTOBAGE', 'IRMJALLGAGE', 'IRALCAGE','ILLICITAGE'], axis=1, inplace=True)
    df.dropna(inplace=True) #war vorher unten falls ihr es nicht selber implementiert habt dann hatte es noch missing values in eurem Datenset (sorry)
    return df


#Feature Filtering (preprocessing)
data_copy = data.copy() #mit NaN

data = rough_filtering(data)

#print(data.isna().sum().sort_values(ascending=False)/data.shape[0])

#data.dropna(inplace=True)

ordinal = []
binary = []
categorical = data.select_dtypes(include=['object','category']).columns.tolist()
for col in categorical:
    if data[col].cat.ordered == True:
            ordinal.append(col)
    else:
        binary.append(col)
binary.remove('UDPYIEM')
#onehotencoding nur für nominal data?
#sonst ordinalencoder?

tests = ['f_classif', 'chi2', 'mututal_info_classif']
#accuracy direkt rausnehmen
#metrics = {'accuracy':[],'precision':[], 'recall':[], 'specificity':[],'f1':[], 'roc_auc':[]}
def eval(y,X,clf,ax,legend_entry='my legendEntry', model_name = 'MyModel'):
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:,1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    #accuracy prolly useneh --> unbalanced also wird eh nicht viel aussagen
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    specificity = tn / (tn + fp)
    fp_rates, tp_rates, _ = roc_curve(y, y_pred_proba)

    roc_auc = auc(fp_rates,tp_rates)
    ax.plot(fp_rates, tp_rates)
    ax.set_title(f'ROC curve for fold {legend_entry}\n({model_name})')
    return [accuracy,precision,recall,specificity,f1,roc_auc], confusion_matrix(y, y_pred)

def selection(X_train,y_train,how,n,what,ax, title='entry'):
    #um von allen den score zu erhalten eifach n= 'all' setzen
    #how möglichkeiten: [chi2,f_classif,mutual_info_classif] also oben bei tests = ...
    UVFS_Selector = SelectKBest(score_func=how, k=n).set_output(transform="pandas")
    X_selected = UVFS_Selector.fit_transform(X_train,y_train)

    if what == 'auswertung':
        return X_selected, UVFS_Selector
    #wenn what == mututal info nicht besten direkt suchen sondern alle werden bewertet --> evt einfacher zum nachher plotten
    if what == 'mutual info':
        mutual_info = mutual_info_classif(X_train,y_train)
        mutual_info = pd.Series(mutual_info, index=X_train.columns).sort_values(ascending=False)
        ax.bar(mutual_info.index, mutual_info)
        ax.set_title(f'Mutual Information scores for fold {title}')
        ax.set_xticks(np.arange(len(mutual_info)),mutual_info.index, rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel('Scores')
        return mutual_info
    if what == 'pic':
        #feature_sel = pd.Series(map(UVFS_Selector.feature_names_in_,UVFS_Selector.scores_),index=UVFS_Selector.feature_names_in_).sort_values(ascending=False)
        feature_sel = pd.Series(UVFS_Selector.scores_,
                                index=UVFS_Selector.feature_names_in_).sort_values(ascending=False)
        #ax.bar(UVFS_Selector.feature_names_in_ , UVFS_Selector.scores_)
        ax.bar(feature_sel.index, feature_sel)
        ax.set_title(f'Feature scores for fold {title}')
        ax.set_xticks(np.arange(len(UVFS_Selector.feature_names_in_)),feature_sel.index, rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel('Scores')

def picture(x,height,features,ax):
    ax.bar(x,height)
    ax.set_xticks(ticks=x,labels=features,rotation=90)
    ax.set_title(input('Title: '))

#fig, axs = plt.subplots(1,4,figsize=(9,4)) um nachher bei allen die roc_auc curve zu machen
"""
model_names = []
for i,ax in enumerate(axs):
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('f'ROC curve for {model_names[i]}')
    add_identity(ax,color='r',ls='--', label = 'random\nclassifier')
    plt.legend(loc='best')
plt.tight_layout()
plt.savefig('figures/roc_curve.png')
"""
#ihr müsst für euer model die parameter (als dictionary) setzen --> z.b. bei random forest max_depth und criterion(gini;entropy)
# mit estimator.get_params() dictionary mit allen möglichen parametern
def hyper(X,y,model,parameters,cv):
    param_grid = parameters
    scoring = {'precision': make_scorer(precision_score),'f1': make_scorer(f1_score),'recall':make_scorer(recall_score)}
    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, refit='f1')

    clf_GS.fit(X,y)
    return clf_GS
    #return [clf_GS.best_estimator_ , clf_GS.best_score_]

def everything(data, model,param, random,what):
    metrics=['accuracy', 'precision', 'recall', 'specificity','F1', 'roc_auc']
    performance = pd.DataFrame(columns=['fold','accuracy','precision','recall','specificity','F1','roc_auc'])
    num_cols = data.select_dtypes(include=['Int64','float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object','category']).columns.tolist()
    cate_cols.remove('UDPYIEM')
    X = data.drop('UDPYIEM',axis=1)
    y= data.UDPYIEM

    ordinal = []
    nominal = []
    for col in cate_cols:
        if data[col].cat.ordered == True:
            ordinal.append(col)
        else:
            nominal.append(col)
    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 0
    tuning = pd.DataFrame(columns=list(param.keys()),index=np.arange(splits))
    model_name = str(input("Which Model used? "))
    #Figures für die Feature selection bilder
    fig,axs = plt.subplots(5,1,figsize=(30,25))
    fig2, axs2 = plt.subplots(1,5,figsize=(16,5))
    zeros = np.zeros((splits,len(X.columns.tolist())),dtype=int)
    selected_features = pd.DataFrame(zeros, index=np.arange(splits),columns = X.columns)
    confusion_matrices = []
    #aus Homework 5
    LR_normcoef = pd.DataFrame(index=X.columns,columns=np.arange(splits))
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        undersampler = RandomUnderSampler(random_state=random)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        num_cols_test = X_test.select_dtypes(include=['Int64', 'float64']).columns.tolist()
        cate_cols_test = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
        ordinal_test = []
        nominal_test = []
        for col in cate_cols_test:
            if X_test[col].cat.ordered == True:
                ordinal_test.append(col)
            else:
                nominal_test.append(col)
        transformer = ColumnTransformer(transformers=[('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan),ordinal),('num',StandardScaler(),num_cols)])
        X_train = transformer.fit_transform(X_train)
        all_columns = ordinal + num_cols
        X_train = pd.DataFrame(X_train, columns=all_columns, index=X_train_copy.index)
        X_train = pd.concat([X_train,X_train_copy[nominal]],axis=1)
        X_test = transformer.transform(X_test)
        all_columns_test = ordinal_test + num_cols_test
        X_test = pd.DataFrame(X_test, columns=all_columns_test, index=X_test_copy.index)
        X_test = pd.concat([X_test,X_test_copy[nominal_test]],axis=1)
        #Da im test set nicht immer alle categories vorhanden sind gibt es im test set auch NaN values --> die muss man zuerst rausnehmen um das test set bezüglich der feature selection anpassen zu können(transform)
        #print(X_test.isna().sum())
        idxmissing= list(X_test.isna().any(axis=1).index[X_test.isna().any(axis=1).values])
        y_test.drop(index=idxmissing,inplace=True)
        X_test.dropna(axis=0, inplace=True)

        #hyperparameter welcher am besten passt finden und in tuning speichern --> am schluss schauen und according zu dem die Hyperparameter setzen:
        tuning.loc[fold,list(param.keys())] = list(map(hyper(X_train,y_train,model,param,5).best_estimator_.get_params().get, list(param.keys())))
        #feature selection (filter method)

        #bild welches für alle 5 folds die Feature scores zeigt --> anschauen und dann auswählen welche am besten sind
        #selection(X_train,y_train,mutual_info_classif,'all','mutual info',axs[fold],title = str(fold+1))
        #Bild von Feature scores Chi2 (Cat input, cat output)
        #selection(X_train[cate_cols],y_train,chi2,'all','pic',axs[fold], title =str(fold+1))
        #Bild von Feature scores f_classif
        #selection(X_train[num_cols],y_train,f_classif,'all','pic',axs[fold], title =str(fold+1))
        #gleiche für mutual_info_classif
        #selection(X_train, y_train, mutual_info_classif, 'all', 'pic', axs[fold], title=str(fold + 1))

        #grobe Feature Selection (Univariate feature selection --> jztmal die 45 besten auswählen --> da in allen 5 folds ungefähr die gleichen gut
        X_train_UVFS, UVFS_Selector = selection(X_train,y_train,mutual_info_classif,45,'auswertung',axs,title='entry')
        clf = model
        #hier embedded Feature selection
        #sel_ = SelectFromModel(clf).set_output(transform="pandas")
        sel_ = SelectFromModel(clf)
        sel_.fit(X_train_UVFS,y_train)
        clf.fit(X_train_UVFS,y_train)


        #hier wenn ich Logistic Regression mache dass es die Feature importance (normalized) abspeichert um nachher ein barplot damit zu machen um die besten zu sehen
        LR = 'LogisticRegression'
        which_model = str(model)
        if LR in which_model:
            #this_LR_coefs = pd.DataFrame(zip(list(UVFS_Selector.get_feature_names_out()), np.transpose(clf.coef_[0])),columns=['features','coef'])
            #LR_normcoef.loc[list(UVFS_Selector.get_feature_names_out()),fold] = this_LR_coefs['coef'].values/this_LR_coefs['coef'].abs().sum()
            this_LR_coefs = pd.DataFrame(zip(X_train_UVFS.columns, np.transpose(clf.coef_[0])),columns=['features', 'coef'])
            LR_normcoef.loc[X_train_UVFS.columns, fold] = this_LR_coefs['coef'].values / this_LR_coefs['coef'].abs().sum()
        #sel_ = SelectFromModel(clf, prefit=True).set_output(transform="pandas")
        #sel_.fit(X_train_UVFS,y_train)

        X_train_L1 = sel_.transform(X_train_UVFS)

        X_test_UVFS = UVFS_Selector.transform(X_test)
        selected_features.loc[fold, list(sel_.get_feature_names_out())] += 1
        #selected_features.loc[fold, X_train_L1.columns] += 1
        clf_L1 = model
        clf_L1.fit(X_train_L1,y_train)
        X_test_L1 = sel_.transform(X_test_UVFS)

        #Performance evaluation
        eval_metrics, cm = eval(y_test,X_test_L1,clf_L1,axs2[fold],legend_entry=str(fold+1),model_name = model_name)
        performance.loc[len(performance),:] = [fold]+eval_metrics
        confusion_matrices.append(cm)

        fold+=1

    if what == 'hyperparameter':
        return tuning.value_counts().head(1)
    if what == 'feature selection':
        plt.tight_layout()
        plt.savefig('figures/Feature_Selection.png')
    if what == 'feature importance LR':
        LR_avg = LR_normcoef.mean(axis=1).sort_values(ascending=False, key=abs)
        error = LR_normcoef.std(axis=1)[LR_avg.index.tolist()]
        plt.figure(figsize=(16,9))
        #plt.clf()
        plt.style.use('ggplot')
        plt.bar(LR_avg.index, LR_avg, yerr=error, align='center', error_kw={'ecolor':'k', 'capsize':3})
        plt.xticks(LR_avg.index,rotation=90)
        plt.title('Normalized feature importance for LR across 5 folds')
        plt.xlabel('Feature')
        plt.ylabel('Normalized feature importance')
        plt.tight_layout()
        plt.savefig('figures/Feature_Importance_LR.png')
    if what == 'selected Features':
        plt.figure(figsize=(16,9))
        plt.style.use('ggplot')
        plt.bar(selected_features.sum(axis=0).sort_values(ascending=False).index, selected_features.sum(axis=0).sort_values(ascending=False))
        plt.xticks(np.arange(len(selected_features.sum(axis=0))), selected_features.sum(axis=0).sort_values(ascending=False).index, rotation=90)
        plt.title('How often these Features were selected in the 5 Folds (with SelectFromModel())')
        plt.xlabel('Features')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('figures/Feature_Counts.png')
        #print(selected_features.sum(axis=0).sort_values(ascending=False))
    if what == 'evaluation':
        for i, ax in enumerate(axs2):
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            add_identity(ax, color='r', ls='--', label = 'random\nclassifier')
            plt.legend(loc='best')

        plt.tight_layout()
        if model_name == 'Logistic Regression':
            plt.savefig('figures/roc_curve_LR.png')
        if model_name == 'KNearest':
            plt.savefig('figures/roc_curve_KN.png')
        #eure model_names :) ...
        if model_name == 'SVM':
            plt.savefig('figures/roc_curve_SVM.png')


        ##if model_name == ...:
        fig_cm, axs_cm = plt.subplots(1, splits, figsize=(15, 5))
        for i, cm in enumerate(confusion_matrices):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs_cm[i])
            axs_cm[i].set_title(f'Confusion Matrix\n({model_name}) for Fold {i + 1}')
            axs_cm[i].set_xlabel('Predicted Label')
            axs_cm[i].set_ylabel('True Label')

        plt.tight_layout()
        if model_name == 'Logistic Regression':
            plt.savefig('figures/confusion_matrices_LR.png')
        if model_name == 'KNearest':
            plt.savefig('figures/confusion_matrices_knearest.png')
        #eure model_names :) ...
        if model_name == 'SVM':
            plt.savefig('figures/confusion_matrices_SVM.png')

        ##if model_name == ...:
        for item in metrics:
            print('{} {}: {:.3f} ± {:.3f}'.format(model_name,item.capitalize(), performance[item].mean(axis=0),performance[item].std(axis=0)))
#bevor implementieren schauen welche parameter auswählen damit man die eingeben kann
#embedded methods für feature selection abhängig von der model wahl --> z.b. bei logistic regression lasso, ridge möglich; bei Random Forest feature importance (alles durch die Funktion SelectFromModel(model([hypterparameters().best_params_]])) = irgendwas --> das dann .fit etc
#SelectFromModel() kann bei allen estimators verwendet werden welche attribute coef_ oder feature_importances_ hat
#parameter ={'C':[0.001,0.01,0.1,1,10]}

#everything(data,LogisticRegression(class_weight='balanced',max_iter=1000),parameter,1,'feature selection')
#Hyperparameter tuning:
#print(everything(data,LogisticRegression(class_weight='balanced',max_iter=1000),parameter,1, 'hyperparameter'))
#C=0.1 am besten
#everything(data,LogisticRegression(class_weight='balanced',max_iter=1000, C=0.1),parameter,1,'feature selection')
#everything(data,LogisticRegression(class_weight='balanced',max_iter=1000, C=0.1),parameter,1,'feature importance LR')
#everything(data,LogisticRegression(class_weight='balanced',max_iter=1000, C=0.1),parameter,1,'selected Features')
#everything(data,LogisticRegression(class_weight='balanced',max_iter=1000, C=0.1),parameter,1,'evaluation')


#svm_parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
#print(everything(data, SVC(probability=True, class_weight='balanced'), svm_parameters, 1, 'hyperparameter').head(1))
#best: C = 0.01, linear, fold 4
#parameter = {'C': [0.01], 'kernel': ['linear']}
#everything(data, SVC(probability=True, class_weight='balanced'), parameter, 1, 'evaluation')

