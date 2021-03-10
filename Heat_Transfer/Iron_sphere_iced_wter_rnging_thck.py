# Code made for Sergio Andrés Díaz Ariza
# 10 March 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 2.2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# Is a sphere we calculate the Perimeter
def As(D,Dx):
    # Here we calculate the perimeter how the mean of the thickness
    Per1 = np.pi*D
    Per2 = np.pi*(D-Dx)
    return (Per2+Per1)/2

def Qcond(T2,T1,Dx,As):
    # k = [W/m*K]
    y = (-80.2)*As*(T2-T1)/Dx
    return y

Range_thick = np.arange(1e-3,0.010,0.0001)
Sup_A = As(0.2,Range_thick)

Q = Qcond(5,0,2e-3,Sup_A)/1000 #[kW]

plt.figure(1)
plt.plot(Range_thick,Q,color='r')
plt.title("Heat Rate Conduction\t$vs$\tRanging Thickness Sphere",fontsize= 18)
plt.xlabel("Thickness\t[$m$]",fontsize= 14)
plt.ylabel("$\dot{Q}_{conduction}$\t$[kW]$",fontsize= 14)
plt.show()