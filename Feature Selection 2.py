import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
print(data.dtypes)
print(data['BLNTAGE'].value_counts())
print(data['BLNTAGE'].isna().sum())