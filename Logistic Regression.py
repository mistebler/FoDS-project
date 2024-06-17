parameter ={'C':[0.001,0.01,0.1,1,10]}
from Feature_Selection_2 import *
#Hyperparameter tuning:
#print(everything(data,LogisticRegression(class_weight='balanced',max_iter=1000),parameter,1, 'hyperparameter'))
#C=0.1 am besten
#In console type in Logistic Regression
everything(data,LogisticRegression(class_weight='balanced',max_iter=1000, C=0.1),parameter,1,'evaluation')

