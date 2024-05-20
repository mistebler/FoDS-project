import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, f_classif, SelectKBest, mutual_info_classif
import scipy.stats as sts
#eif drop_na??
data = pd.read_csv('drug-use-health/data_new.csv', index_col=0)
def cleaning(data):
    data['UDPYIEM'] = pd.Categorical(data['UDPYIEM'])
    none_991 = ['IRCIGAGE', 'IRCDUAGE', 'IRCGRAGE', 'IRSMKLSSTRY', 'IRALCAGE', 'IRMJAGE', 'IRTOBAGE','IRMJALLGAGE']
    data['BLNTAGE'] = data['BLNTAGE'].where(data['BLNTAGE'] < 70, None).astype('Int64')
    binary = ['CIGFLAG','CGRFLAG','PIPFLAG', 'SMKLSSFLAG', 'TOBFLAG', 'ALCFLAG','MRJFLAG','DCIGMON','FUCIG18','FUCIG21', 'FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','DEPNDALC','DEPNDMRJ', 'ABUSEALC','ABUSEMRJ','ABODALC','ABODMRJ','CDUFLAG','CDCGMO','BNGDRKMON','HVYDRKMON','FUCD218','FUCD221','DNICNSP','CIGDLYMO','CIG100LF','PIPE30DY','BLNTNOMJ','BLNTEVER']
    ordinal = ['CIG30TPE','ALCYDAYS','MRJYDAYS','CIGMDAYS','CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','BNGDRMDAYS','CIGAVGD', 'AGEALC','AGETOB','AGEMRJ']
    for i in none_991:
        data[i] = data[i].replace({991:None, 993:None, 999:None})
    for i in ordinal:
        if i == 'CIGAVGD':
            data[i] = pd.Categorical(data[i], ordered=True)
        else:
            data[i] = pd.Categorical(data[i].astype('Int64'), ordered=True)
    for i in binary:
        data[i] = pd.Categorical(data[i].astype('Int64'))

    convert = ['ALCUS30D','IRCIGAGE', 'IRCDUAGE', 'IRCGRAGE', 'IRSMKLSSTRY', 'IRALCAGE','IRMJAGE', 'BLNTAGE', 'BLNT30DY','IRTOBAGE','IRMJALLGAGE']
    for i in convert:
        data[i] = data[i].astype('Int64')
    data.drop(['IRTOBAGE', 'IRMJALLGAGE', 'IRALCAGE'], axis=1, inplace=True)
    return data

#print(data.select_dtypes(include='float64').columns.tolist())
"""
anschauen = []

for i in data.columns.tolist():
    if i in data.select_dtypes(include='Int64').columns:
        if data.loc[data[i]>90,i].shape[0]!=0:
            anschauen.append(i)
    elif i in data.select_dtypes(include='float64').columns:
        if data.loc[data[i]>90.0,i].shape[0]!=0:
            anschauen.append(i)
    if i in ordinal or i in binary:
        if data.loc[data[i].astype('Float64')>90.0,i].shape[0]!=0:
            anschauen.append(i)
print(anschauen)
"""
#for i in data.select_dtypes(include='category').columns.tolist():
    #print(data[i].value_counts())
#for col in data.columns:
        #if data[col].dtype == "category":
            #print(f"Column {col} ordered? {data[col].cat.ordered}")

#Feature Continuous --> Anova
#Feature Categorical --> Chi-Square
#print(data.isna().sum().sort_values(ascending=False)/data.shape[0])
data = cleaning(data)
#Grobe Feature selection (meisten Nan values)
plt.figure(figsize=(14,6))
plt.bar(data.isna().sum().sort_values(ascending=False).index, data.isna().sum().sort_values(ascending=False)/data.shape[0])
plt.xticks(data.isna().sum().sort_values(ascending=False).index, rotation=90)
plt.ylabel('% missing values')
plt.xlabel('Features')
plt.title('Amount of missing values')
plt.tight_layout()
plt.savefig('figures/missing_values.png')
#print(IRMJALLGAGE.isna().sum()/data.shape[0])
pd.set_option("display.max_columns", data.shape[1])
data.drop(['IRSMKLSSTRY','IRCGRAGE','IRCIGAGE','BLNTAGE','IRMJAGE','IRCDUAGE'],axis=1,inplace=True)


#Feature Filtering (preprocessing)
data_copy = data.copy() #mit NaN
"""
def decide(data):
    if input('Alter wichtiger? (y/n): ').lower() == 'y':
        data.drop(['CIGFLAG','CGRFLAG','PIPFLAG','SMKLSSFLAG','TOBFLAG','ALCFLAG','MRJFLAG'],inplace=True,axis=1)
    else:
        data.drop(['IRTOBAGE','IRMJALLGAGE','IRALCAGE'],axis=1,inplace=True)
    return data
data = decide(data)
"""
print(data.isna().sum().sort_values(ascending=False))

data.dropna(inplace=True)
#print(data.TOBFLAG.value_counts())
#print(data_copy.TOBFLAG.value_counts())
#print(data.UDPYIEM.value_counts())
#print(data_copy.UDPYIEM.value_counts())

#print(data_evt.shape)
#data_evt2 = data.dropna(subset=['IRMJALLGAGE','IRTOBAGE','IRALCAGE'])
#print(data_evt2.shape)
#print(data_evt2.isna().sum().sort_values(ascending=False))
#print(data.isna().sum().sort_values(ascending=False))
num_cols = data.select_dtypes(include=['Int64','float64']).columns.tolist()
cate_cols = data.select_dtypes(include=['object','category']).columns.tolist()
cate_cols.remove('UDPYIEM')
X = data.drop('UDPYIEM',axis=1)
y = data['UDPYIEM']
statistic = pd.DataFrame(index = ['F-Statistic','p-value','Chi2 statistic','Ranksum'],columns = X.columns)
normal = []
alpha = 0.05 / data.shape[1]
for col in num_cols:
    group1 = X.loc[y.values== 1, col]
    group2 = X.loc[y.values== 0, col]
    p_val1 = sts.normaltest(group1).pvalue
    p_val2 = sts.normaltest(group2).pvalue
    if p_val1 < alpha or p_val2 < alpha:
        statistic.loc['Ranksum',col] = sts.ranksums(group1, group2).statistic
        statistic.loc['p-value',col] = sts.ranksums(group1, group2).pvalue
        #nicht normal --> ranksum test
    else:
        normal.append(col)

if len(normal) != 0:
    f_statistic, p_values = f_classif(X[normal],y)
    statistic.loc['F-Statistic',normal] = f_statistic
    statistic.loc['p-value',normal] = p_values
chi2_stats, p_values = chi2(X[cate_cols],y)
statistic.loc['Chi2 statistic',cate_cols] = chi2_stats
statistic.loc['p-value', cate_cols] = p_values
print(statistic.loc['p-value',:].sort_values(ascending=False))

plt.figure(figsize=(9,4))
plt.bar(statistic.loc['p-value',:].sort_values(ascending=False).index, statistic.loc['p-value',:].sort_values(ascending=False))
plt.xticks(statistic.loc['p-value',:].sort_values(ascending=False).index, rotation=90)
plt.ylabel('p-value')
plt.xlabel('Features')
plt.title('p-values of the Features \n(for (rough) feature filtering)')
plt.axhline(alpha, linestyle='--', color='r', label='Significance level')
plt.legend()
plt.tight_layout()
plt.savefig('figures/p-value-distribution.png')


def selection(data, model):

    num_cols = data.select_dtypes(include=['Int64','float64']).columns.tolist()
    cate_cols = data.select_dtypes(include=['object','category']).columns.tolist()
    cate_cols.remove('UDPYIEM')
    data_enc = pd.get_dummies(data, columns=cate_cols, drop_first=True)
    X = data_enc.drop('UDPYIEM',axis=1)
    y= data.UDPYIEM
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

    #X_train = scaler.fit_transform(X_train[num_cols])
    #X_test = scaler.transform(X_test[num_cols])
    #UVFS_Selector = SelectKBest(f_classif, k=)
    splits = 5
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)
    fold = 0
    data_mutual_info = pd.DataFrame(columns = X.columns, index = np.arange(splits))
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        #statistical testing the training features
        data_mutual_info.loc[fold,:] = mutual_info_classif(X_train,y_train)

        fold+=1
