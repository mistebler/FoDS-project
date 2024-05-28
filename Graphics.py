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
from Feature_Selection_2 import *

data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)

data = cleaning(data)
data =rough_filtering(data)

"""
#Pie plots
df_subset = data.iloc[:, :25]

# Define the figure and axes
fig, axes = plt.subplots(5, 5, figsize=(18, 12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Create a pie chart for each variable in the subset
for i, column in enumerate(df_subset.columns):
    counts = df_subset[column].value_counts()
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%')
    axes[i].set_title(f'Pie chart of {column}')

# Adjust layout
plt.tight_layout()
plt.show()

#Histogram
df_subset = data[["IRALCFY","IRMJFY"]]

# Define the figure and axes
fig, axes = plt.subplots(2, 1, figsize=(10, 8))


# Create a bar plot for each variable in the subset
for i, column in enumerate(df_subset.columns):
    sns.histplot(data[column], bins=10, ax=axes[i], kde=True)
    axes[i].set_title(f'Bar plot of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.show()