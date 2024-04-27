import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data.csv', low_memory=False)
pd.set_option("display.max_columns", data.shape[1])
print(data.shape)
#alle columns mit fragen zu tobacco, marijuahna, alcohol raussuchen und neue csv datei machen