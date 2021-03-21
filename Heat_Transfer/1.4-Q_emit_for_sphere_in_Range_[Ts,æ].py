# Code made for Sergio Andrés Díaz Ariza
# 06 March 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 1.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

def As(D):  #Superficial Area of cilinder [m]
    return np.pi*D**2

def Qemit(e,As,Ts): #Power emitted by the package
    # e = emissivity
    # õ = 5.670e-8 [W/m^2 * K^4]
    Q = e * 5.670e-8 * As *Ts**4
    return Q

A_sup = As(0.1) # Surface area with a especific diameter
Rnge_temp_K = np.arange(40+273.15,85+273.15,1) #[K] Careful the units must be in Kelvin for Q function
Q = Qemit(0.25,A_sup,Rnge_temp_K)
plt.figure(1)
plt.plot(np.arange(40,85,1),Q,'c')
plt.title("Power Dissipation\n$vs$\nTemperature Surface")
plt.xlabel("Temperature Surface\t$[^{\circ}C]$ ")
plt.ylabel("$\dot{Q}_{emited}$\t$[W]$")

Rnge_emiss = np.linspace(0.20,0.30,num=45) # Range of values for emissivity
Q2 = Qemit(Rnge_emiss,A_sup,Rnge_temp_K)
fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(Rnge_emiss,Q2,'c')
ax1.set_xlim(0.20,0.30)
ax2.set_xlim(40,85)
ax1.set_xticks(np.linspace(0.20, 0.30, num=11))
ax2.set_xticks(np.linspace(40, 85, num=11))
plt.title("Power Dissipation\t$vs$\tTemperature Surface\t$&$\tEmissivity",fontsize= 18)
ax2.set_xlabel("Temperature Surface\t$[^{\circ}C]$ ",fontsize= 14)
ax1.set_xlabel("Emissivity\t$[\epsilon]$ ",fontsize= 14)
ax1.set_ylabel("$\dot{Q}_{emited}$\t$[W]$",fontsize= 14)
plt.show()

