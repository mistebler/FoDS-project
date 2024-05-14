import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
#print(data.dtypes)
binary = ['CIGFLAG','CGRFLAG','PIPFLAG', 'SMKLSSFLAG', 'TOBFLAG', 'ALCFLAG','MRJFLAG','DCIGMON','FUCIG18','FUCIG21', 'FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','DEPNDALC','DEPNDMRJ', 'ABUSEALC','ABUSEMRJ','ABODALC','ABODMRJ','CDUFLAG','CDCGMO','BNGDRKMON','HVYDRKMON','FUCD218','FUCD221','DNICNSP','CIGDLYMO','CIG100LF','PIPE30DY','BLNTNOMJ','BLNTEVER']
ordinal = ['CIG30TPE','ALCYDAYS','MRJYDAYS','CIGMDAYS','CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','BNGDRMDAYS','CIGAVGD']

for i in ordinal:
    data[i] = pd.Categorical(data[i], ordered=True)
for i in binary:
    data[i] = pd.Categorical(data[i])

print(data.select_dtypes(include='float64').columns)
print(data['IRCIGFM'].value_counts())