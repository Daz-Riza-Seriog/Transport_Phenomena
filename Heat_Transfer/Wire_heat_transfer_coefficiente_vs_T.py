# Code made for Sergio Andrés Díaz Ariza
# 06 March 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 1.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

def As(R,h): # Here we determine the superficial Area
    # R = [m]
    # h = [m]
    A_sup = 2*np.pi*R*h + 2*np.pi*R**2
    return A_sup

def h_conv(Q_conv,Area,T_s,T_surr):
    # Q_conv = [W]
    # T_s, T_surr = [K] or [Celsius]
    hconv = Q_conv/(Area*(T_s-T_surr))
    return hconv

Ar_sup = As(1e-3,2.1) # Put the values of Radius and Length fot thi calculus
heat_coefficient1 = h_conv(330,Ar_sup,180,20) # Determine the coefficient

Rnge_Temp = np.arange(100,300,1) # [Celsius] Range of values of Temperature
heat_coefficient = h_conv(330,Ar_sup,Rnge_Temp,20)

# Plot and look
plt.figure(1)
plt.plot(Rnge_Temp,heat_coefficient,color="r")
plt.title("Heat Transfer Coefficient\n$vs$\nTemperature Range")
plt.ylabel("$h_{convection}$\t$[W/m*K]$")
plt.xlabel("$T$\t$[K]$")
plt.show()
