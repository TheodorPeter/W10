import numpy as np
import matplotlib.pyplot as plt
import wichtige_Funktionen as fct
import scipy.optimize as opt
from scipy import integrate


directory = 'W10_Messprotokoll.TXT'

Tkalt = np.loadtxt(directory, delimiter='\t', skiprows=116, max_rows=11, usecols=1)
t1 = np.loadtxt(directory, delimiter='\t', skiprows=116, max_rows=11, usecols=0)
Twarm = np.loadtxt(directory, delimiter='\t', skiprows=116, max_rows=11, usecols=2)
t2 = np.loadtxt(directory, delimiter='\t', skiprows=128, max_rows=26, usecols=0)
Tmisch1 = np.loadtxt(directory, delimiter='\t', skiprows=128, max_rows=26, usecols=1)
t3 = np.loadtxt(directory, delimiter='\t', skiprows=155, max_rows=11, usecols=0)
Tmisch2 = np.loadtxt(directory, delimiter='\t', skiprows=155, max_rows=11, usecols=1)
t12 = np.empty(37)
t23 = np.empty(37)
for x in range(11):
    t12[x]=t1[x]
    t23[26+x] = t3[x]
for x in range(26):
    t12[11+x] = t2[x]
    t23[x]=t2[x]


#Ausgleichsgeraden Parameter
pauskalt = np.polyfit(t1, Tkalt, 1)
pauswarm = np.polyfit(t1, Twarm, 1)
pausmisch = np.polyfit(t3, Tmisch2, 1)

def ausgleichkalt(x):
    return (pauskalt[0] * x + pauskalt[1])

def ausgleichwarm(x):
    return (pauswarm[0] * x + pauswarm[1])
def ausgleichmisch(x):
    return(pausmisch[0] * x + pausmisch[1])
#Grenzgeraden:
grenz = fct.steigunggrenz(t3, Tmisch2)
def grenzgerade1(x):
    return fct.grenzgerade(x, Tmisch2, grenz[0])

def grenzgerade2(x):
    return fct.grenzgerade(x, Tmisch2, grenz[1])

#polyfit des mittleren stücks
poly = np.polyfit(t2, Tmisch1, 8)


#plt.show()

def mitte(x):
    return poly[0]*x**8 + poly[1]*x**7 + poly[2]*x**6 + poly[3]*x**5 + poly[4]*x**4 + poly[5]*x**3 + poly[6]*x**2 + poly[7]*x + poly[8]


#Funktionen für die rechte Flächen:
def arear(x):
    return (ausgleichmisch(x) - mitte(x))
def areal(x):
    return (mitte(x) - ausgleichkalt(x))
def arear1(x):
   return(grenzgerade1(x)-mitte(x))
def arear2(x):
   return(grenzgerade2(x)-mitte(x))

y = np.empty(1)
y1 = np.empty(1)
y2 = np.empty(1)

mini = np.empty(1)
mini1 = np.empty(1)
mini2 = np.empty(1)
for x in range(1, 1000, 1):
    if(x == 1):
        mini = 50
        mini2 = 50
        mini1 = 50
    dif = np.abs(integrate.quad(areal, t1[-1], t1[-1] + x*(t2[-1]-t1[-1])/1000)[0] - integrate.quad(arear,  t1[-1] + x*(t2[-1]-t1[-1])/1000, t2[-1])[0])
    dif1 = np.abs(integrate.quad(areal, t1[-1], t1[-1] + x*(t2[-1]-t1[-1])/1000)[0] - integrate.quad(arear1,  t1[-1] + x*(t2[-1]-t1[-1])/1000, t2[-1])[0])
    dif2 = np.abs(integrate.quad(areal, t1[-1], t1[-1] + x*(t2[-1]-t1[-1])/1000)[0] - integrate.quad(arear2,  t1[-1] + x*(t2[-1]-t1[-1])/1000, t2[-1])[0])

    if(dif < mini):
        mini = dif
        y=x
    if (dif1 < mini1):
        mini1 = dif1
        y1 = x

    if (dif2 < mini2):
        mini2 = dif2
        y2 = x

tstern = t1[-1] + y*(t2[-1]-t1[-1])/1000
tstern2 = t1[-1] + y1*(t2[-1]-t1[-1])/1000
tstern3 = t1[-1] + y2*(t2[-1]-t1[-1])/1000


#Beginn des Plots
plt.plot(t1, Tkalt, 'k.',label='Messwerte')
plt.plot(t2, Tmisch1, 'k.')
plt.plot(t3, Tmisch2, 'k.')
plt.plot(t1, Twarm, 'k.')


#Ausgleichsgeraden
plt.plot(t12, ausgleichwarm(t12), 'r', linewidth=0.8, label = 'Ausgleichsgerade T_warm')
plt.plot(t12, ausgleichkalt(t12), 'b', label = 'Ausgleichsgerade T_kalt')
plt.plot(t23, ausgleichmisch(t23), 'g', label = 'Ausgleichsgerade T_misch')
#senkrechte geraden
plt.vlines(x = tstern, ymin = ausgleichkalt(tstern), ymax = ausgleichwarm(tstern),colors= 'k', label='t*')
plt.plot(tstern, ausgleichwarm(tstern), 'rs', label = 'T_warm')
plt.plot(tstern, ausgleichmisch(tstern), 'gs', label = 'T_misch')
plt.plot(tstern, ausgleichkalt(tstern), 'bs', label ='T_kalt' )
plt.title('Zwickelabgleich Aufgabe 1')
plt.grid()
plt.legend()
plt.show()

mh = 0.4765-0.2775
mk = 0.4586-0.2421
Tkalt = ausgleichkalt(tstern)
Twarm = ausgleichwarm(tstern)
Tmisch = ausgleichmisch(tstern)
print("tstern:", tstern)
print("Tkalt:", ausgleichkalt(tstern))
print("Twarm:", ausgleichwarm(tstern))
print("Tmisch:", ausgleichmisch(tstern))
w = mh*(Twarm-Tmisch)/(Tmisch-Tkalt)-mk
print("Wasserwert ohne Gruppennummer:", w)
wgruppe = w + 0.0013
print("Wasswerwert mit Gruppennummer", wgruppe)



#Aufgabe 2
cw = 4184
mwalu = 0.6857 - 0.2421
mwku = 0.670 - 0.2403


t2 = np.arange(1, 41, 1)
Toku = np.loadtxt(directory, skiprows=6, max_rows=40, usecols=1)
Tuku = np.loadtxt(directory, skiprows=6, max_rows=40, usecols=2)
Toal = np.loadtxt(directory, skiprows=63, max_rows=40, usecols=1)
Tual = np.loadtxt(directory, skiprows=63, max_rows=40, usecols=2)

grenzku = fct.steigunggrenz(t2[2:], Tuku[2:])
patuku = np.polyfit(t2[2:], Tuku[2:], 1)

plt.plot(t2, Tuku, 'k.', label = 'Messwerte')
plt.plot(t2[2:],fct.printpolynom(t2[2:], patuku) ,'r-', label = 'Ausgleichsgerade')
plt.plot(t2[2:], fct.grenzgerade(t2[2:], Tuku[2:], grenzku[0]),'r--', label = 'Grenzgeraden' )
plt.plot(t2[2:], fct.grenzgerade(t2[2:], Tuku[2:], grenzku[1]),'r--' )
plt.title("untere Temperatur von Kupfer")
plt.legend()
plt.grid()
plt.xlabel("t in min")
plt.ylabel("T in °C")
plt.show()



grenzalu = fct.steigunggrenz(t2[2:], Tual[2:])
patalu = np.polyfit(t2[2:], Tual[2:], 1)

plt.plot(t2, Tual, 'k.', label = 'Messwerte')
plt.plot(t2[2:],fct.printpolynom(t2[2:], patalu) ,'r-', label = 'Ausgleichsgerade')
plt.plot(t2[2:], fct.grenzgerade(t2[2:], Tual[2:], grenzalu[0]),'r--', label = 'Grenzgeraden' )
plt.plot(t2[2:], fct.grenzgerade(t2[2:], Tual[2:], grenzalu[1]),'r--' )
plt.xlabel("t in min")
plt.ylabel("T in °C")
plt.legend()
plt.grid()
plt.title("untere Temperatur von Aluminium")
plt.show()

pku = cw*(mwku +wgruppe)*patuku[0]/60
palu = cw*(mwalu + wgruppe)*patalu[0]/60
print("Steigung kupfer:", patuku[0], "in SI Einheiten:", patuku[0]/60)
print("Steigung der Grenzgeraden von Kupfer:", grenzku, "in SI Einheiten:", grenzku/60)
print("Steigung Alu:", patalu[0], "in SI Einheiten:", patalu[0]/60)
print("Steigung der Grenzgeraden von Alu:", grenzalu, "in SI Einheiten:", grenzalu/60)
print("Wärmeleistung Kupfer:", pku)
print("Wärmeleistung Aluminium", palu)
#Aufgabe 3


plt.plot(t2, Tual,'.r-',label = 'T Unten')
plt.plot(t2, Toal,'.g-',label = 'T Oben')
plt.plot(t2,(Toal-Tual),'.b-',label = 'Delta T')
plt.title('Temperaturen für Aluminium')
plt.grid()
plt.xlabel("t in min")
plt.ylabel("T in °C")
plt.legend()
plt.show()

plt.plot(t2,Tuku,'.r-',label = 'T Unten')
plt.plot(t2,Toku,'.g-',label = 'T Oben')
plt.plot(t2,(Toku-Tuku),'.b-',label = 'Delta T')
plt.title('Temperaturen für Kupfer')
plt.grid()
plt.xlabel("t in min")
plt.ylabel("T in °C")
plt.legend()
plt.show()

deltalu = fct.mittelwert((Toal-Tual)[26:])
deltku = fct.mittelwert((Toku-Tuku)[26:])
print("Temperaturdifferenz Alu Stationär:", fct.mittelwert((Toal-Tual)[26:]))
print("Fehler Tempreraturdifferenz Alu stationär:", fct.mittelwertfehler((Toal-Tual)[26:]))
print("Temperaturdifferenz Kupfer Stationär:", fct.mittelwert((Toku-Tuku)[26:]))
print("Fehler Temperaturdifferenz Kupfer stationär:", fct.mittelwertfehler((Toku-Tuku)[26:]))

#Aufgabe 4
delxalu = 0.3
delxku = 0.3
ralu = 0.0125
rku = 0.0125

lambdaku = pku * delxku /(np.pi * rku **2 * deltku)
lambdaalu = palu * delxalu /(np.pi * ralu**2 * deltalu)

print("Lambda kupfer:", lambdaku)
print("Lambda Alu:", lambdaalu)