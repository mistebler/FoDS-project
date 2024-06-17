svm_parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
parameter = {'C': [0.01], 'kernel': ['linear']}
from Feature_Selection_2 import *
#print(everything(data, SVC(probability=True, class_weight='balanced'), svm_parameters, 1, 'hyperparameter').head(1))
#best: C = 0.01, linear, fold 4
everything(data, SVC(probability=True, class_weight='balanced',C=0.01, kernel='linear'), parameter, 1, 'evaluation')
