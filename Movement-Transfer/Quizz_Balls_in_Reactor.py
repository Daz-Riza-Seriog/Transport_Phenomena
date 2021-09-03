# Code made for Sergio Andrés Díaz Ariza
# 18 August 2021
# License MIT
# Transport Phenomena: Quiz Ball in Reactor

import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()

# Parameters
D_hail = 0.005  # [m]
A_hail = (np.pi * (D_hail ** 2)) / 4  # Area reflect
miu = 0.0013  # [Kg/m*s]
rho = 1.1  # [Kg/m^3]
rho_hail = 1250  # [Kg/m^3]
g = 9.87  # [m/s^2]

# Volume hail
V_hail = (4 * np.pi * ((D_hail / 2) ** 3))/3

# Weigth of the hail
W_hail = V_hail * rho_hail * g
print(W_hail)

# Bouguer force
F_b = rho * V_hail * g
print(F_b)

# Balance
F_d = W_hail - F_b
print(F_d)

# Reynolds Eq:
def Reynolds(rho, vel, Diam, miu):
    Rey = (rho * Diam * vel) / miu
    return Rey


# Find the Velocity:
def Velocity(F, Cd, A, rho):
    vel = np.sqrt((2 * F) / (Cd * A * rho))
    return vel


# Determine Cd number with Graphic and Reynolds number
Rey_can = Reynolds(rho, 5.467, D_hail, miu)

vel = Velocity(F_d, 2.5, A_hail, rho)

print("Find The Reynolds Number:\t{:e}".format(Rey_can))
print("Velocity find:\t{:.3f}".format(vel))

print("\n--- %s seconds ---" % (time.time() - start_time))
