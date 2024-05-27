from code import *
from Feature_Selection_2 import *
from sklearn.svm import SVC
from sklearn.metrics import make_scorer

#integrate other kernel for non-linear!!!!
#should i do feature selection in svm_analysis function after hyperparameter tunin?
#everything function und svm_analysis function zusammenführen
def eval(y,X,clf,ax,legend_entry='my legendEntry'):
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:,1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=1)
    recall = recall_score(y, y_pred, zero_division=1)
    f1 = f1_score(y, y_pred, zero_division=1)
    specificity = tn / (tn + fp)
    fp_rates, tp_rates,_ = roc_curve(y, y_pred_proba) #hier fehlt bodenstrich

    roc_auc = auc(fp_rates,tp_rates)
    ax.plot(fp_rates, tp_rates, label=f'Classifier fold {legend_entry} ')
    return [accuracy, precision,recall,specificity,f1,roc_auc]

def hypertuning(X,y,param_grid,model,cv):
    scoring = {'accuracy': 'accuracy', 'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score)}

    clf_GS = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=cv, verbose=1)
    clf_GS.fit(X, y)
    # nachher hyptertuning().etc nehmen
    #gibt den besten estimator an --> z.B. um y_prediction zu bekommen hyptertuning(...).predict(X) angeben
    return clf_GS.best_estimator_

def everything(data, model, param_grid, random, ax, tune_hyperparameters=False):
    performance = pd.DataFrame(columns=['fold','clf','accuracy','precision','recall','specificity','F1','roc_auc'])
    num_cols = data.select_dtypes(include=['Int64','float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object','category']).columns.tolist()
    cate_cols.remove('UDPYIEM')
    data_enc = pd.get_dummies(data, columns=cate_cols, drop_first=True)
    X = data_enc.drop('UDPYIEM',axis=1)
    y= data.UDPYIEM

    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random)
    fold = 0

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # hyperparameter tuning
        """if tune_hyperparameters:
            model = hypertuning(X_train, y_train, param_grid, model, cv=splits)"""


        model.fit(X_train, y_train)

        fold_metrics = eval(y_test, X_test, model, ax, legend_entry=str(fold))

        fold_performance = pd.DataFrame([{'fold': fold,'clf': str(model),'accuracy': fold_metrics[0],'precision': fold_metrics[1],
            'recall': fold_metrics[2],'specificity': fold_metrics[3],'F1': fold_metrics[4],'roc_auc': fold_metrics[5]}])

        performance = pd.concat([performance, fold_performance], ignore_index=True)

        fold += 1
    return performance
def svm_analysis(data,random_state=42):
    fig, ax = plt.subplots()
    param_linear = {'kernel': 'linear', 'probability': True, 'random_state': random_state}
    kernels = ['rbf', 'poly', 'sigmoid']  # List of kernels for non-linear SVM
    performance = pd.DataFrame()

    #Evaluate linear kernel
    linear_svm = SVC(**param_linear)
    linear_performance = everything(data, linear_svm, None, random_state, ax)
    performance = pd.concat([performance, linear_performance], ignore_index=True)

    #non-linear
    for kernel in kernels:
        param_grid = {'kernel': [kernel], 'C': [0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}

        nonlinear_performance = everything(data, SVC(probability=True), param_grid, random_state, ax,tune_hyperparameters=True)
        nonlinear_performance['kernel'] = kernel
        performance = pd.concat([performance, nonlinear_performance], ignore_index=True) #combine performance data

    ax.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig('figures/SVM_VIC.png')
    plt.show()

    return performance


data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
cleaned_data = cleaning(data)
filtered_data = rough_filtering(cleaned_data)
filtered_data.dropna(inplace=True)

performance = svm_analysis(filtered_data)
print(performance)