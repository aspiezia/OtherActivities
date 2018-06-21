import json
import requests

## FUNCTION DEFINED IN ORDER TO CALCULATE THE GAIN ##
def gainCalculator(initial, final, step, gain, rate):
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                if step == 0:
                    currency1=currency[0]
                    currency2=currency[i]
                if step == 1:
                    currency1=currency[i]
                    currency2=currency[j]
                if step == 2:
                    currency1=currency[j]
                    currency2=currency[k]
                if step == 3:
                    currency1=currency[k]
                    currency2=currency[0]
                if initial == currency1 and final == currency2:
                    gain[i][j][k] = gain[i][j][k]*rate
    return gain

## USER PARAMETERS ##
initialCurrency = 'JPY'
verbose = False # Set it to True if you want to check all the possible changes
userFile = False # Set it to True if you want to use a private file, otherwise it will download the updated rates from "https://fx.priceonomics.com/v1/rates/"
if initialCurrency!='EUR' and initialCurrency!='USD' and initialCurrency!='JPY' and initialCurrency!='BTC':
    print("The currency %s is not considered in our exercise. Please use one of the following: ['EUR', 'USD', 'JPY', 'BTC']" %(initialCurrency))
    quit()

## OPEN THE FILE ##
if userFile:
    file = open('rates.json').read()
    data = json.loads(file)
else:
    data = json.loads(requests.get("https://fx.priceonomics.com/v1/rates/").text)

## INITIALIZE THE PARAMETERS ##
currency = ['EUR', 'USD', 'JPY', 'BTC']
if initialCurrency == 'USD':
    currency = ['USD', 'EUR', 'JPY', 'BTC']
if initialCurrency == 'JPY':
    currency = ['JPY', 'USD', 'EUR', 'BTC']
if initialCurrency == 'BTC':
    currency = ['BTC', 'USD', 'JPY', 'EUR']
gain = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

## CALCULATE THE GAIN ##
i=0
for key, value in data.items():
    initial,final = (list(data.keys())[i]).split('_')
    rate =  float(list(data.values())[i])
    for step in range(0,4):
        gain = gainCalculator(initial, final, step, gain, rate)  
    i=i+1

## PRINT THE RESULTS ##
GAIN=0
A=''
B=''
C=''
for j in range(0,4):
    for k in range(0,4):
        for l in range(0,4):
            if verbose:
                if currency[0]!=currency[j] and currency[j]!=currency[k] and currency[k]!=currency[l] and currency[l]!=currency[0]:
                    print("The gain for %s->%s->%s->%s->%s is %0.2f%%" %(currency[0], currency[j], currency[k], currency[l], currency[0], (100*(gain[j][k][l]-1))))
                if currency[0]==currency[j] and currency[j]!=currency[k] and currency[k]!=currency[l] and currency[l]!=currency[0]:
                    print("The gain for %s->%s->%s->%s is %0.2f%%" %(currency[0], currency[k], currency[l], currency[0], (100*(gain[j][k][l]-1))))
                if currency[0]==currency[j] and currency[j]==currency[k] and currency[k]!=currency[l] and currency[l]!=currency[0]:
                    print("The gain for %s->%s->%s is %0.2f%%" %(currency[0], currency[l], currency[0], (100*(gain[j][k][l]-1))))
            if (100*(gain[j][k][l]-1))>GAIN:
                GAIN = (100*(gain[j][k][l]-1))
                A=currency[j]
                B=currency[k]
                C=currency[l]
print("")
print("The best gain is for the change %s->%s->%s->%s->%s = %0.2f%%" %(currency[0], A, B, C, currency[0], GAIN))
        
    



