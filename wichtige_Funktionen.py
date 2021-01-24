import numpy as np
import math

#Fixed Version
def mittelwert(x):
    return(np.sum(x)/np.size(x))

def standardabweichung(x):
    return np.sqrt(np.sum((x-mittelwert(x))**2)/(np.size(x)-1))

def mittelwertfehler(x):
    return standardabweichung(x)/np.sqrt(np.size(x))

def grenzgerade(x, y, steigung):
    return (mittelwert(y) + (steigung)*(x-mittelwert(x)) )

def xaxabgrenz(x, y, steigung):
    return (-mittelwert(y)/steigung + mittelwert(x))

def yaxabgrenz(x, y, steigung):
    return(mittelwert(y) - steigung * mittelwert(x))

def sortiere(x, y):
    xsort = np.sort(x)
    yneu = np.zeros(np.size(y))
    for z in range(np.size(x)):
        for h in range(np.size(x)):
            if xsort[z] == x[h]:
                yneu[z] = y[h]
    return (xsort, yneu)


def steigunggrenz (x, y):
    x, y = sortiere(x, y)
    param = np.polyfit(x, y, 1)
    steigung = param[0]
    n = np.size(x)*0.317
    if (math.ceil(n) -n) > (n - math.floor(n)):
        n = math.floor(n)
    else:
        n = math.ceil(n)
    maxab = np.abs(0.9 * steigung)
    min = np.size(x)
    mittel = np.array([mittelwert(x), mittelwert(y)])
    m = 0
    l = 0
    for z in range(np.size(x)-1):
        if(x[z] > mittel[0]):
            l = z
            break
    for z in range(0, 1001, 1):
        counter = 0
        for k in range(l):
            a = mittel[1] + (steigung-(maxab - z * maxab/1000)) * (x[k]-mittel[0])
            b = mittel[1] + (steigung+(maxab - z * maxab/1000)) * (x[k]-mittel[0])
            if (y[k] > a or y[k] < b):
                    counter = counter + 1

        for k in range(l, np.size(x)-1, 1):
            a = mittel[1] + (steigung-(maxab - z * maxab/1000)) * (x[k] - mittel[0])
            b = mittel[1] + (steigung+(maxab - z * maxab/1000)) * (x[k] - mittel[0])
            if (y[k] < a or y[k] > b):
                    counter = counter + 1
        #print(counter)
        if (np.abs(counter - n) < min):
            min = np.abs(counter - n)
            m = z
    return np.array([steigung -(maxab - m * maxab/1000), steigung + (maxab - m * maxab/1000)])
