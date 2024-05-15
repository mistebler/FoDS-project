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
#print(data['UDPYIEM'].value_counts())
data['UDPYIEM'] = pd.Categorical(data['UDPYIEM'])
#data['UDPYIEM'] = data['UDPYIEM'].cat.rename_categories({0:'No',1:'Yes'})
#print(data.dtypes)
#print(data['UDPYIEM'].value_counts())
binary = ['CIGFLAG','CGRFLAG','PIPFLAG', 'SMKLSSFLAG', 'TOBFLAG', 'ALCFLAG','MRJFLAG','DCIGMON','FUCIG18','FUCIG21', 'FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','DEPNDALC','DEPNDMRJ', 'ABUSEALC','ABUSEMRJ','ABODALC','ABODMRJ','CDUFLAG','CDCGMO','BNGDRKMON','HVYDRKMON','FUCD218','FUCD221','DNICNSP','CIGDLYMO','CIG100LF','PIPE30DY','BLNTNOMJ','BLNTEVER']
change0 = ['FUCIG18','FUCIG21', 'FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','FUCD218','FUCD221','BLNTEVER']
ordinal = ['CIG30TPE','ALCYDAYS','MRJYDAYS','CIGMDAYS','CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','BNGDRMDAYS','CIGAVGD']
change91_0 = ['IRCIGFM','IRCGRFM','IRSMKLSS30N','IRALCFM','IRALCBNG30D','IRMJFM','BLNT30DY']
change991_0 = ['IRALCFY','IRMJFY','ALCUS30D']
change5_0 = ['CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','BNGDRMDAYS']
change6_0 = ['ALCYDAYS', 'MRJYDAYS','CIGMDAYS']
none_991 = ['IRCIGAGE','IRCDUAGE','IRCGRAGE','IRSMKLSSTRY','IRALCAGE','IRMJAGE']
data['CIGAVGD'] = data['CIGAVGD'].fillna(0)
data['PIPE30DY'] = data['PIPE30DY'].replace({1:1,2:0,91:0,94: None,97: None, 98: None})
data['CIGDLYMO'] = data['CIGDLYMO'].replace({1:1,2:0,5:1,91:0,94: None,97: None})
data['CIG100LF'] = data['CIG100LF'].replace({1:1,2:0,3:1,5:1,91:0,94: None,97: None})
data['BLNTNOMJ'] = data['BLNTNOMJ'].replace({1:1,2:0,5:1,14:1,24:1,91:0,93:0,98:None})
data['CIG30TPE'] = data['CIG30TPE'].replace({1:2,2:1,91:0,93:0,94:None,97:None,98:None})
data['BLNTEVER'] = data['BLNTEVER'].replace({2:0,4:0,11:1,94:None,97:None,98:None})
for i in binary:
    data[i] = pd.Categorical(data[i])
for i in change0:
    data[i] = data[i].cat.rename_categories({1:1,2:0})
for i in ordinal:
    data[i] = pd.Categorical(data[i], ordered=True)
for i in change5_0:
    data[i] = data[i].cat.rename_categories({1:1,2:2,3:3,4:4,5:0})
for i in change6_0:
    data[i] = data[i].cat.rename_categories({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6:0})
for i in change91_0:
    data[i] = data[i].replace({91: 0, 93:0})
for i in change991_0:
    data[i] = data[i].replace({991:0, 993:0})
for i in none_991:
    data[i] = data[i].replace({991:None, 993:None})
#data['ALCUS30D'].replace(91 | 93,0,inplace=True)
#data['BLNT30DY'].replace(91 | 93,0,inplace=True)

data['BLNT30DY'] = data['BLNT30DY'].replace({94:None,97:None,98:None})
data['ALCUS30D']= data['ALCUS30D'].replace({975:5, 985:None, 994:None, 997:None, 998:None})
#data['CIG30TPE'] = data['CIG30TPE'].replace({1:2,2:1,91:0,93:0,94:None,97:None,98:None})
#data['PIPE30DY'] = data['PIPE30DY'].cat.rename_categories({1:1,2:0,91:0})
#data['CIGDLYMO'] = data['CIGDLYMO'].cat.rename_categories({1:1,2:0,5:1,91:0})
#data['CIG100LF'] = data['CIG100LF'].cat.rename_categories({1:1,2:0,3:1,5:1,91:0})
#data['BLNTNOMJ'] = data['BLNTNOMJ'].cat.rename_categories({1:1,2:0,5:1,14:1,24:1,91:0})
#data['CIG30TPE'] = data['CIG30TPE'].cat.rename_categories({1:2,2:1,91:0,93:0})
"""
data['PIPE30DY'] = data['PIPE30DY'].replace({1:1,2:0,91:0,94: None,97: None, 98: None})
data['CIGDLYMO'] = data['CIGDLYMO'].replace({1:1,2:0,5:1,91:0,94: None,97: None})
data['CIG100LF'] = data['CIG100LF'].replace({1:1,2:0,3:1,5:1,91:0,94: None,97: None})
data['BLNTNOMJ'] = data['BLNTNOMJ'].replace({1:1,2:0,5:1,14:1,24:1,91:0,93:0,98:None})
"""
#data['BLNTAGE'] = data['BLNTAGE'].astype(int)

#print(data['BLNTAGE'].value_counts())
data['BLNTAGE'] = data['BLNTAGE'].where(data['BLNTAGE'] < 70,None).astype('Int64')



data.to_csv('drug-use-health/data_new.csv', na_rep=None)
#pd.set_option("display.max_rows", data.shape[0])

#print(data.select_dtypes(include='float64').columns)
#print(data['ALCUS30D'].value_counts())
#print(data['BLNTAGE'].value_counts())

