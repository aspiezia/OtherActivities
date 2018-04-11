import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble.partial_dependence import plot_partial_dependence


################### Creating Dataset ###################
InitialDataframe = pd.read_csv('anafile_challenge_170522.csv', names=['Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'DaysTrying', 'CyclesTrying', 'ExitStatus', 'AnovCycles'], skiprows=1)
SkimmedDataframe = InitialDataframe.loc[(InitialDataframe['ExitStatus']==' Pregnant')]
Pill          = SkimmedDataframe['Pill']
NCbefore      = SkimmedDataframe['NCbefore']
FPlength      = SkimmedDataframe['FPlength']
CycleVar      = SkimmedDataframe['CycleVar']
_, PILL       = np.unique(Pill,       return_inverse=True)
_, NCBEFORE   = np.unique(NCbefore,   return_inverse=True)
_, FPLENGHT   = np.unique(FPlength,   return_inverse=True)
_, CYCLEVAR   = np.unique(CycleVar,   return_inverse=True)
FinalDataframe = pd.DataFrame(data={'CyclesTrying': SkimmedDataframe['CyclesTrying'], 'TempLogFreq': SkimmedDataframe['TempLogFreq'], 'SexLogFreq': SkimmedDataframe['SexLogFreq'], 'AnovCycles': SkimmedDataframe['AnovCycles'], 'NCbefore': NCBEFORE, 'CycleVar': CYCLEVAR, 'Pill': PILL, 'FPlength': FPLENGHT})
y = SkimmedDataframe['DaysTrying']
X_train, X_test, y_train, y_test = train_test_split(FinalDataframe, y, test_size=0.1)
########################################################


################### Fitting TheModel ###################
params = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_split': 5,'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
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
plt.savefig('Ranking.pdf')
plt.show()

##### Plot Training Deviance ####
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)
plt.title('')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-', label='Training Set')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set')
plt.legend(loc='upper right')
plt.xlabel('Number of estimators')
plt.ylabel('Error')
plt.savefig('DeviancePlot.pdf')
plt.show()
