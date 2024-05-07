import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug-use-health/data.csv', low_memory=False, index_col=0)
pd.set_option("display.max_columns", data.shape[1])
#print(data.shape)
IRCDUAGE = pd.read_csv('drug-use-health/IRCDUAGE.csv', low_memory=False, index_col=0)
UDPYIEM =pd.read_csv('drug-use-health/UDPYIEM.csv', low_memory=False, index_col=0)
data_copy = data.copy()
columns = data.filter(regex = r'(IRTBL|CRAVE|CRAGP|INCTL|AVOID|FN|PLANE|RNOUT|REG|NMCHG|SVLHR|INFLU|NOINF|INCRS|SATIS|LOTMR|WAKE|ROUT|RGDY|RGWK|RGNM|LOTTM|GTOVR|LIMIT|KPLMT|NDMOR|LSEFX|CUT|WD2SX|WDSMT|EMOPB|EMCTD|PHLPB|PHCTD|LSACT|SERPB|PDANG|LAWTR|FMFPB|FMCTD|DRVIN|NDT|TX|HP|SN|YE|STND|PR|YFL|FRD|HLTIN|NDFLT|MED|MX|MM|WRK|MFU|RSK|DIFOBT|RSN|NUMDKPM|DIFGET|OFRSM|WILYR|AUAL|JOBNG)', axis=1).columns.tolist()
data.drop(columns,axis=1, inplace=True)
data = pd.concat([data,data_copy['MJONLYFLAG']], axis =1)
data = pd.concat([data,IRCDUAGE], axis =1)
data = pd.concat([data,UDPYIEM], axis =1)
data.to_csv('drug-use-health/data_clean.csv')
#print(data.columns.tolist())
data_neu = data[['CIG30AV','CIG30TPE','CIGDLYMO','CIG100LF','PIPE30DY','ALCUS30D','IRALCFY','IRMJFY','IRCIGFM','IRCGRFM','IRSMKLSS30N','IRALCFM','IRALCBNG30D','IRMJFM','IRCIGAGE','IRCDUAGE','IRCGRAGE','IRSMKLSSTRY','IRALCAGE','IRMJAGE','CIGFLAG','CGRFLAG','PIPFLAG','SMKLSSFLAG','TOBFLAG','ALCFLAG','MRJFLAG','DCIGMON','ALCYDAYS','MRJYDAYS','CIGMDAYS','CGRMDAYS','SMKLSMDAYS','ALCMDAYS','MRJMDAYS','CIGAVGD','CIGAVGM','FUCIG18','FUCIG21','FUCGR18','FUCGR21','FUSMKLSS18','FUSMKLSS21','FUALC18','FUALC21','FUMJ18','FUMJ21','BLNTEVER','BLNTAGE','BLNT30DY','BLNTNOMJ','DEPNDALC','DEPNDMRJ','DPPYILLALC','ABUSEALC','ABUSEMRJ','ABODALC','ABODMRJ','CDUFLAG','CDCGMO','BNGDRKMON','HVYDRKMON','BNGDRMDAYS','FUCD218','FUCD221','DNICNSP','UDPYIEM']]
#print(data_neu.head(5))
data_neu.to_csv('drug-use-health/data_clean.csv')
