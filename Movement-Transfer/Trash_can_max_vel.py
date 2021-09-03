# Code made for Sergio Andrés Díaz Ariza
# 25 July 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 2.1

import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()


# Parameters
W_can = 4 #[Kg]
D_can = 0.5  # [m]
L_tower = 0.980  # [m]
A_tower = L_tower / D_can  # Area reflect
miu = 1.849e-5  # [Kg/m*s]
g = 9.87 # [m/s^2]

# Tor Balance
F_d = W_can*g/(L_tower/2)

L_D = L_tower/D_can
print("L/D for find Cd:\t{:.3f}".format(L_D))

# Lineal Regression for density air:
def Density_air(T_amb):
    y = (-4e-3) * T_amb + 1.29
    return y

# Find the Velocity:
def Velocity(F,Cd,A,rho):
    vel = np.sqrt(2*F/(Cd*A*rho))
    return vel

# Reynolds Eq:
def Reynolds(rho, vel, Diam, miu):
    Rey = (rho * Diam * vel) / miu
    return Rey


# Determine Cd number with Graphic and Reynolds number

rho = Density_air(25)
vel = Velocity(F_d,0.8,A_tower,rho)
Rey_can = Reynolds(rho, vel, D_can, miu)

print("Find The Reynolds Number:\t{:e}".format(Rey_can))
print("Velocity find:\t{:.3f}".format(vel))
# Second Iteration of the Cd

print("\n--- %s seconds ---" % (time.time() - start_time))