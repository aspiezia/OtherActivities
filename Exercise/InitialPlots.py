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
FinalDataframe = pd.DataFrame(data={'Country': COUNTRY, 'Age': Age, 'NumBMI': NumBMI, 'Pill': PILL, 'NCbefore': NCBEFORE, 'FPlength': FPLENGHT, 'Weight': Weight, 'CycleVar': CYCLEVAR, 'TempLogFreq': TempLogFreq, 'SexLogFreq': SexLogFreq, 'CyclesTrying': CyclesTrying, 'ExitStatus': EXITSTATUS, 'AnovCycles': AnovCycles, 'DaysTrying': DaysTrying})
########################################################


################## Plotting Variables ##################
plots_tot = 14
plots_name = ['Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'DaysTrying', 'CyclesTrying', 'ExitStatus', 'AnovCycles']
BIN =  [88,60,80,3,2,3,100,2,100,100,100,40,3,12]
XMAX = [87,59,80,2,1,2,300,1,1  ,1,  999,39,2,11]
YMAX = [20000,1800,3000,14000,18000,14000,3000,12000,600,4500,2000,5000,10000,18000]
i=0
while i < plots_tot:
    print('%i. Plotting %s' % (i,plots_name[i]))
    if i == 0:
        fig,ax=plt.subplots(figsize=(20, 10))
        ax.set_xticks(range(len(unique1)))
        ax.set_xticklabels(unique1)
    elif i == 3:
        fig,ax=plt.subplots()
        ax.set_xticks(range(len(unique2)))
        ax.set_xticklabels(unique2)
    elif i == 4:
        fig,ax=plt.subplots()
        ax.set_xticks(range(len(unique3)))
        ax.set_xticklabels(unique3)
    elif i == 5:
        fig,ax=plt.subplots()
        ax.set_xticks(range(len(unique4)))
        ax.set_xticklabels(unique4)
    elif i == 7:
        fig,ax=plt.subplots()
        ax.set_xticks(range(len(unique5)))
        ax.set_xticklabels(unique5)
    elif i == 12:
        fig,ax=plt.subplots()
        ax.set_xticks(range(len(unique6)))
        ax.set_xticklabels(unique6)
    plt.figure(i+1)
    plt.hist(FinalDataframe[plots_name[i]], bins=BIN[i], normed=0, facecolor='green')
    plt.xlabel(plots_name[i])
    plt.ylabel('Women')
    plt.title('')
    plt.axis([0, XMAX[i], 0, YMAX[i]])
    plt.grid(True)
    if i == 0:
        plt.xticks(fontsize=8)
    plt.savefig(plots_name[i]+'.pdf')
    i = i+1
########################################################


############# Plotting Correlation Matrix ##############
plt.figure(15)#, figsize=(40, 20))
corr_matrix = np.corrcoef([DaysTrying, COUNTRY, Age, NumBMI, PILL, NCBEFORE, FPLENGHT, Weight, CYCLEVAR, TempLogFreq, SexLogFreq, CyclesTrying, AnovCycles])
plt.matshow(corr_matrix, cmap=plt.cm.get_cmap('coolwarm'), vmin=-1)
for i in range(0,13):
    for j in range(0,13):
        plt.text(j, i, '%.2f' % corr_matrix[i,j], ha='center', va='center',fontsize=7)
labels = ['DaysTrying', 'Country', 'Age', 'NumBMI', 'Pill', 'NCbefore', 'FPlength', 'Weight', 'CycleVar', 'TempLogFreq', 'SexLogFreq', 'CyclesTrying', 'AnovCycles']
plt.xticks(range(0,13),labels, rotation='vertical', fontsize=8)
plt.yticks(range(0,13),labels, fontsize=8)
plt.colorbar()
plt.savefig('CorrelationMatrix.pdf')
#plt.show()
########################################################


#### Printing Correlations DaysTrying-OtherVariable ####
print('')
print('Correlation DaysTrying-Country      = %.2f' % np.corrcoef(DaysTrying, COUNTRY)[0][1])
print('Correlation DaysTrying-Age          = %.2f' % np.corrcoef(DaysTrying, Age)[0][1])
print('Correlation DaysTrying-NumBMI       = %.2f' % np.corrcoef(DaysTrying, NumBMI)[0][1])
print('Correlation DaysTrying-PILL         = %.2f' % np.corrcoef(DaysTrying, PILL)[0][1])
print('Correlation DaysTrying-NCBEFORE     = %.2f' % np.corrcoef(DaysTrying, NCBEFORE)[0][1])
print('Correlation DaysTrying-FPLENGHT     = %.2f' % np.corrcoef(DaysTrying, FPLENGHT)[0][1])
print('Correlation DaysTrying-Weight       = %.2f' % np.corrcoef(DaysTrying, Weight)[0][1])
print('Correlation DaysTrying-CYCLEVAR     = %.2f' % np.corrcoef(DaysTrying, CYCLEVAR)[0][1])
print('Correlation DaysTrying-TempLogFreq  = %.2f' % np.corrcoef(DaysTrying, TempLogFreq)[0][1])
print('Correlation DaysTrying-SexLogFreq   = %.2f' % np.corrcoef(DaysTrying, SexLogFreq)[0][1])
print('Correlation DaysTrying-CyclesTrying = %.2f' % np.corrcoef(DaysTrying, CyclesTrying)[0][1])
print('Correlation DaysTrying-AnovCycles   = %.2f' % np.corrcoef(DaysTrying, AnovCycles)[0][1])  
print('')
