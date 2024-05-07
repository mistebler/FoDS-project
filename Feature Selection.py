import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data_clean.csv', low_memory=False, index_col=0)
print(data.shape)
pd.set_option("display.max_columns", data.shape[1])
#print(data.head(5))
print(data.dtypes)
print(data['UDPYIEM'].value_counts())