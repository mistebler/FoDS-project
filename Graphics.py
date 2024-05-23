import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, f_classif, SelectKBest, mutual_info_classif, VarianceThreshold, SelectFromModel
import scipy.stats as sts
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,  precision_recall_curve
from 'Feature Selection 2.py' import *

data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
