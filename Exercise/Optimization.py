import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import GridSearchCV


################### Creating Dataset ###################
InitialDataframe = pd.read_csv('anafile_challenge_170522.csv', names=['Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'DaysTrying', 'CyclesTrying', 'ExitStatus', 'AnovCycles'], skiprows=1)
SkimmedDataframe = InitialDataframe.loc[(InitialDataframe['ExitStatus']==' Pregnant')]
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
FinalDataframe = pd.DataFrame(data={'Country': COUNTRY, 'Age': Age, 'NumBMI': NumBMI, 'Pill': PILL, 'NCbefore': NCBEFORE, 'FPlength': FPLENGHT, 'Weight': Weight, 'CycleVar': CYCLEVAR, 'TempLogFreq': TempLogFreq, 'SexLogFreq': SexLogFreq, 'CyclesTrying': CyclesTrying, 'AnovCycles': AnovCycles})
y = SkimmedDataframe['DaysTrying']
X_train, X_test, y_train, y_test = train_test_split(FinalDataframe, y, test_size=0.2)
########################################################


############## Optimization GridSearchCV ###############
param_grid = {'max_depth': [4, 6],'min_samples_leaf': [3, 5, 9, 17, 25],'learning_rate': [0.1, 0.05, 0.02, 0.01, 0.005],}
optimization = ensemble.GradientBoostingRegressor(n_estimators=500)
gs_cv = GridSearchCV(optimization, param_grid, n_jobs=4).fit(X_train, y_train)
print(gs_cv.best_params_)
########################################################
