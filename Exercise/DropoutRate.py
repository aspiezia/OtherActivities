import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble.partial_dependence import plot_partial_dependence


################### Creating Dataset ###################
InitialDataframe = pd.read_csv('anafile_challenge_170522.csv', names=['Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'DaysTrying', 'CyclesTrying', 'ExitStatus', 'AnovCycles'], skiprows=1)
SkimmedDataframe = InitialDataframe.loc[((InitialDataframe['ExitStatus']==' Pregnant') | (InitialDataframe['ExitStatus']==' Dropout')) & (InitialDataframe['NumBMI']>0)]
Country       = SkimmedDataframe['Country']
Age           = SkimmedDataframe['Age']
NumBMI        = SkimmedDataframe['NumBMI']
Pill          = SkimmedDataframe['Pill']
NCbefore      = SkimmedDataframe['NCbefore']
FPlength      = SkimmedDataframe['FPlength']
Weight        = SkimmedDataframe['Weight']
CycleVar      = SkimmedDataframe['CycleVar']
TempLogFreq   = SkimmedDataframe['TempLogFreq']
SexLogFreq    = SkimmedDataframe['SexLogFreq']
DaysTrying    = SkimmedDataframe['DaysTrying']
CyclesTrying  = SkimmedDataframe['CyclesTrying']
ExitStatus    = SkimmedDataframe['ExitStatus']
AnovCycles    = SkimmedDataframe['AnovCycles']
unique1, COUNTRY    = np.unique(Country,    return_inverse=True)
unique2, PILL       = np.unique(Pill,       return_inverse=True)
unique3, NCBEFORE   = np.unique(NCbefore,   return_inverse=True)
unique4, FPLENGHT   = np.unique(FPlength,   return_inverse=True)
unique5, CYCLEVAR   = np.unique(CycleVar,   return_inverse=True)
unique6, EXITSTATUS = np.unique(ExitStatus, return_inverse=True)
FinalDataframe = pd.DataFrame(data={'Country': COUNTRY, 'Age': Age, 'NumBMI': NumBMI, 'Pill': PILL, 'NCbefore': NCBEFORE, 'FPlength': FPLENGHT, 'Weight': Weight, 'CycleVar': CYCLEVAR, 'TempLogFreq': TempLogFreq, 'SexLogFreq': SexLogFreq, 'CyclesTrying': CyclesTrying, 'AnovCycles': AnovCycles, 'DaysTrying': DaysTrying})
FinalDataframe2 = pd.DataFrame(data={'ExitStatus': EXITSTATUS})
y = FinalDataframe2['ExitStatus']
X_train, X_test, y_train, y_test = train_test_split(FinalDataframe, y, test_size=0.2, random_state=123456)
########################################################


################### Fitting TheModel ###################
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 5,'learning_rate': 0.05, 'loss': 'deviance'}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train,y_train)
mae = mean_absolute_error(y_test, clf.predict(X_test))
mse = mean_squared_error(y_test, clf.predict(X_test))
acc = clf.score(X_test, y_test)
print('')
print('ACC: %.4f' % acc)
print("MSE: %.4f" % mse)
print('MAE: %.4f' % mae)
print('')

#### Plot feature importance ####
indices = np.argsort(clf.feature_importances_)
names = list(X_train.columns.values)
plt.barh(np.arange(len(names)), clf.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
plt.xlabel('Relative importance')
plt.yticks(fontsize=10)
plt.savefig('RankingDropout.pdf')
plt.show()
