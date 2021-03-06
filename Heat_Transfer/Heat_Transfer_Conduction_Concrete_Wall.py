# Code made for Sergio Andrés Díaz Ariza
# 05 March 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 1.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# Fourier Heat Transfer Equation
def Qcond(Kc,T1,T2,As,Dx):
    # Kc = [W/m*K]
    DT = T2-T1
    z = -Kc*As*(DT/Dx) #[W]
    return z #[W]

##########################################
# Heat Loss ranging Temperatures outside #
# and changing thermal conductivities    #
##########################################

Range_temp = np.arange(-15,38,1) #Range of temperatures to evalute

y = Qcond(0.85,25,Range_temp,20,0.30)
y1 = Qcond(0.75,25,Range_temp,20,0.30)
y2 = Qcond(1.25,25,Range_temp,20,0.30)

plt.figure(1)
plt.plot(Range_temp,y,color='r',label="$Kc$ 0.85")
plt.plot(Range_temp,y1,color='b',label="$Kc$ 0.75")
plt.plot(Range_temp,y2,color='g',label="$Kc$ 1.25")
plt.xticks(np.arange(-15,38,5))
plt.title("Heat Rate Conduction\t$vs$\tRanging Temperature Wall Concrete",fontsize= 18)
plt.xlabel("Temperature\t[$K$]",fontsize= 14)
plt.ylabel("$\dot{Q}_{conduction}$\t$[W]$",fontsize= 14)
plt.legend()

##########################################
# Heat Loss ranging Temperatures outside #
# and changing Thickness wall            #
##########################################

Range_thick = np.linspace(0.10,0.50,num=53) #Range of thickness to evalute

y = Qcond(0.85,25,Range_temp,20,Range_thick)
y1 = Qcond(0.75,25,Range_temp,20,Range_thick)
y2 = Qcond(1.25,25,Range_temp,20,Range_thick)

fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(Range_thick,y,color='r',label="$Kc$ 0.85")
ax1.plot(Range_thick,y1,color='b',label="$Kc$ 0.75")
ax1.plot(Range_thick,y2,color='g',label="$Kc$ 1.25")
ax1.set_xlim(0.10,0.50)
ax2.set_xlim(-15,38)
ax2.set_xticks(np.linspace(-15, 38, num=9))
ax2.set_xlabel("Temperature\t$[K]$",fontsize= 14)
ax1.set_xlabel("Thickness Wall\t$[m]$",fontsize= 14)
ax1.set_ylabel("$\.{Q}_{conduction}$\t${W}$",fontsize= 14)
plt.title("Heat Rate Conduction $vs$ Ranging Thickness $&$ Temperatures in Wall Concrete",fontsize= 18)
ax1.legend()
plt.show()
