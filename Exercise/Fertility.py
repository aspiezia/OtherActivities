import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence


################### Creating Dataset ###################
InitialDataframe = pd.read_csv('anafile_challenge_170522.csv', names=['Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'DaysTrying', 'CyclesTrying', 'ExitStatus', 'AnovCycles'], skiprows=1)
SkimmedDataframe = InitialDataframe.loc[((InitialDataframe['ExitStatus']==' Pregnant') | (InitialDataframe['ExitStatus']==' Dropout') & (InitialDataframe['DaysTrying']>100))]
Country       = SkimmedDataframe['Country']
NCbefore      = SkimmedDataframe['NCbefore']
Pill          = SkimmedDataframe['Pill']
FPlength      = SkimmedDataframe['FPlength']
CycleVar      = SkimmedDataframe['CycleVar']
ExitStatus    = SkimmedDataframe['ExitStatus']
_, NCBEFORE   = np.unique(NCbefore,   return_inverse=True)
_, PILL       = np.unique(Pill,       return_inverse=True)
_, COUNTRY    = np.unique(Country,    return_inverse=True)
_, FPLENGHT   = np.unique(FPlength,   return_inverse=True)
_, CYCLEVAR   = np.unique(CycleVar,   return_inverse=True)
_, EXITSTATUS = np.unique(ExitStatus, return_inverse=True)
FinalDataframe = pd.DataFrame(data={'Age': SkimmedDataframe['Age'], 'AnovCycles': SkimmedDataframe['AnovCycles'], 'CycleVar': CYCLEVAR, 'FPlength': FPLENGHT, 'Weight': SkimmedDataframe['Weight'], 'Pill': PILL})
FinalDataframe2 = pd.DataFrame(data={'ExitStatus': EXITSTATUS})
y = FinalDataframe2['ExitStatus']
X_train, X_test, y_train, y_test = train_test_split(FinalDataframe, y, test_size=0.1)
########################################################


################### Fitting TheModel ###################
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 9,'learning_rate': 0.01, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train,y_train)
pre = precision_score(y_test, clf.predict(X_test))
f1s = f1_score(y_test, clf.predict(X_test))
acc = clf.score(X_test, y_test)
print('')
print('ACC: %.4f' % acc)
print("F1S: %.4f" % f1s)
print('PRE: %.4f' % pre)
print('')

#### Plot feature importance ####
indices = np.argsort(clf.feature_importances_)
names = list(X_train.columns.values)
plt.barh(np.arange(len(names)), clf.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
plt.xlabel('Relative importance')
plt.yticks(fontsize=10)
plt.savefig('Ranking_Fertility.pdf')
#plt.show()

