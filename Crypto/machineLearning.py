import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence

#################### Creating Dataset ###################
InitialDF = pd.read_hdf('coinmarketcap.hdf')
InitialDF = InitialDF.replace('', 0, regex=True)
tokens = ['namecoin', 'litecoin', 'bitcoin', 'peercoin', 'novacoin', 'feathercoin', 'terracoin', 'bitbar', 'worldcoin', 'digitalcoin', 'goldcoin', 'primecoin', 'megacoin', 'anoncoin', 'ripple', 'freicoin', 'ixcoin', 'bullion', 'infinitecoin', 'quark', 'phoenixcoin', 'zetacoin', 'fastcoin', 'tagcoin', 'argentum', 'florincoin', 'casinocoin', 'nxt', 'deutsche-emark', 'sexcoin']
SkimmedDataframe = InitialDF.loc[(InitialDF['currency'].isin(tokens))]
unique, CURRENCY = np.unique(SkimmedDataframe['currency'], return_inverse=True)
FinalDataframe = pd.DataFrame(data={'Close':SkimmedDataframe['Close'],'Volume':SkimmedDataframe['Volume'],'MktCap':SkimmedDataframe['MktCap'],'Diff':((SkimmedDataframe['Close']-SkimmedDataframe['Open'])/SkimmedDataframe['Open']),'Norm':SkimmedDataframe['MktCap']/SkimmedDataframe['Close']})
FinalDataframe2 = pd.DataFrame(data={'currency': CURRENCY})
y = FinalDataframe2['currency']
X_train, X_test, y_train, y_test = train_test_split(FinalDataframe, y, test_size=0.2)
########################################################


################### Fitting TheModel ###################
params = {'n_estimators': 2000, 'max_depth': 4, 'min_samples_split': 9,'learning_rate': 0.01, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train,y_train)
acc = clf.score(X_test, y_test)
print('')
print('ACC: %.4f' % acc)
print('')
########################################################


############### Plot feature importance ################
indices = np.argsort(clf.feature_importances_)
names = list(X_train.columns.values)
plt.barh(np.arange(len(names)), clf.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
plt.xlabel('Relative importance')
plt.yticks(fontsize=10)
plt.savefig('Ranking.pdf')
########################################################
