# Code made for Sergio Andrés Díaz Ariza
# 05 March 2021
# License MIT
# Transport Phenomena: Python Program - Assignment 1.2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

###################################################
# Hot air at 80°C is blown over a 2-m x 4-m flat  #
# surface at 30°C. If the average convection heat #
# transfer coefficient is 55 W/m2·K               #
###################################################

def Qconv(h,As,Tsurr,Ts):
    # h = [W/m^2 * K]
    z = h*As*(Tsurr-Ts) #[W]
    return z

# Range of values of heat transfer coefficient
Range_h = np.arange(20,100,1)

y = Qconv(Range_h,8,80,30)/1000 #[kW]

plt.figure(1)
plt.plot(Range_h,y,color="r")
plt.xlabel("$h_{convection}$\t$[W/m^2]$")
plt.ylabel("$\dot{Q}_{convection}$\t[W]")
plt.title("$\dot{Q}_{convection}$\t$[kW]$\t$vs$\t$h_c$ ")
plt.show()