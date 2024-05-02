import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data.csv', low_memory=False, index_col=0)
pd.set_option("display.max_columns", data.shape[1])
#print(data.shape)
IRCDUAGE = pd.read_csv('drug-use-health/IRCDUAGE.csv', low_memory=False, index_col=0)
data_copy = data.copy()
columns = data.filter(regex = r'(IRTBL|CRAVE|CRAGP|INCTL|AVOID|FN|PLANE|RNOUT|REG|NMCHG|SVLHR|INFLU|NOINF|INCRS|SATIS|LOTMR|WAKE|ROUT|RGDY|RGWK|RGNM|LOTTM|GTOVR|LIMIT|KPLMT|NDMOR|LSEFX|CUT|WD2SX|WDSMT|EMOPB|EMCTD|PHLPB|PHCTD|LSACT|SERPB|PDANG|LAWTR|FMFPB|FMCTD|DRVIN|NDT|TX|HP|SN|YE|STND|PR|YFL|FRD|HLTIN|NDFLT|MED|MX|MM|WRK|MFU|RSK|DIFOBT|RSN|NUMDKPM|DIFGET|OFRSM|WILYR|AUAL|JOBNG)', axis=1).columns.tolist()
data.drop(columns,axis=1, inplace=True)
data = pd.concat([data,data_copy['MJONLYFLAG']], axis =1)
data = pd.concat([data,IRCDUAGE], axis =1)
print(data.columns.tolist())

