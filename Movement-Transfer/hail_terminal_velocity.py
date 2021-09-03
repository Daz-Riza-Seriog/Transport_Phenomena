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
D_hail = 12 / (10 * 100)  # [m]
A_hail = (np.pi * (D_hail ** 2)) / 4  # Area reflect
miu = 1.778e-5  # [Kg/m*s]
rho = 1.246  # [Kg/m^3]
rho_hail = 0.9 * 1e6 / 1000  # [Kg/m^3]
g = 9.87  # [m/s^2]

# Volume hail
V_hail = (4 * np.pi * ((D_hail / 2) ** 3))/3

# Weigth of the hail
W_hail = V_hail * rho_hail * g

# Bouguer force
F_b = rho * V_hail * g

# Balance
F_d = W_hail - F_b


# Reynolds Eq:
def Reynolds(rho, vel, Diam, miu):
    Rey = (rho * Diam * vel) / miu
    return Rey


# Find the Velocity:
def Velocity(F, Cd, A, rho):
    vel = np.sqrt((2 * F) / (Cd * A * rho))
    return vel


# Determine Cd number with Graphic and Reynolds number
Rey_can = Reynolds(rho, 16.9, D_hail, miu)

vel = Velocity(F_d, 0.4, A_hail, rho)

print("Find The Reynolds Number:\t{:e}".format(Rey_can))
print("Velocity find:\t{:.3f}".format(vel))

print("\n--- %s seconds ---" % (time.time() - start_time))
