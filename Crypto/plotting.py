import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#################### Creating Dataset ###################
InitialDF = pd.read_hdf('coinmarketcap.hdf')
InitialDF = InitialDF.replace('', 0, regex=True)
tokens = ['namecoin', 'litecoin', 'bitcoin', 'peercoin', 'novacoin', 'feathercoin', 'terracoin', 'bitbar', 'worldcoin', 'digitalcoin', 'goldcoin', 'primecoin', 'megacoin', 'anoncoin', 'ripple', 'freicoin', 'ixcoin', 'bullion', 'infinitecoin', 'quark', 'phoenixcoin', 'zetacoin', 'fastcoin', 'tagcoin', 'argentum', 'florincoin', 'casinocoin', 'nxt', 'deutsche-emark', 'sexcoin']
colors = ['black',    'red',      'blue',    'green',    'pink',     'yellow',      'purple',    'salmon', 'olive',     'teal',        'cyan',     'navy',      'brown',    'indigo',   'skyblue','lawngreen','aqua',   'dimgrey', 'gold',         'orange','wheat',       'khaki',    'beige',    'peru',    'coral',    'lime',       'crimson',    'tan', 'sienna',         'lightgreen']
#########################################################

def plotter(figure, name, MIN, MAX):
    plt.figure(0)
    for i in range(0,len(tokens)):
        df = InitialDF.loc[(InitialDF['currency']==tokens[i])]
        if name == 'Norm':
            if (df['MktCap']/df['Close']).max() > MIN and (df['MktCap']/df['Close']).max() < MAX:
                plt.plot(df['Date'], df['MktCap']/df['Close'], color=colors[i], label=tokens[i])
        elif name == 'Diff':
            if (abs(df['High']-df['Low'])/df['High']).max() > MIN and (abs(df['High']-df['Low'])/df['High']).max() < MAX:
                plt.plot(df['Date'], (abs(df['High']-df['Low'])/df['High']), color=colors[i], label=tokens[i])
        else:
            if df[name].max() > MIN and df[name].max() < MAX:
                plt.plot(df['Date'], df[name], color=colors[i], label=tokens[i])
    plt.xticks(rotation='vertical')
    plt.ylabel(name)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(name+'_'+str(figure)+'.pdf')
    plt.gcf().clear()
    
for i in range(6,12):
    plotter(i,'MktCap',pow(10,i),pow(10,i+1))

for i in range(3,12,3):
    plotter(i,'Volume',pow(10,i),pow(10,i+3))

for i in range(0,5):
    plotter(i,'Close',pow(10,i),pow(10,i+1))

for i in range(5,12):
    plotter(i,'Norm',pow(10,i),pow(10,i+1))

for i in range(0,2):
    plotter(i,'Diff',i*0.7,(i+1)*0.7)

