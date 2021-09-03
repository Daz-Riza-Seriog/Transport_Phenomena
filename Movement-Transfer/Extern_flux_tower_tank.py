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
D_tank = 4  # [m]
D_tower = 1.2  # [m]
A_tank = np.pi * (D_tank ** 2) / 4  # Area reflect
L_tower = 10  # [m]
A_tower = L_tower / D_tower  # Areareflect

Vel = 130 * (1000 / 3600)  # [m/s]
miu = 1.849e-5  # [Kg/m*s]


# Lineal Regression for density air:
def Density_air(T_amb):
    y = (-4e-3) * T_amb + 1.29
    return y


# Reynolds Eq:
def Reynolds(rho, vel, Diam, miu):
    Rey = (rho * Diam * vel) / miu
    return Rey


# Determine Cd number with Graphic and Reynolds number

rho = Density_air(25)
Rey_tank = Reynolds(rho, Vel, D_tank, miu)
Rey_tower = Reynolds(rho, Vel, D_tower, miu)

print("Number of Reynolds tank:\t{:e}".format(Rey_tank))
print("Number of Reynolds tower:\t{:e}".format(Rey_tower))

C_d_tank = 0.3  # Value extract from the table
C_d_tower = 0.2  # Value extract from the table


# Determine the force over the tank
def F_drag(C_d, Area, rho, vel):
    F = C_d * Area * rho * (vel ** 2) / 2
    return F


F_tank = F_drag(C_d_tank, A_tank, rho, Vel)
F_tower = F_drag(C_d_tower, A_tower, rho, Vel)

# Determine the Torsion moment
T_m_tank = (L_tower + (D_tank / 2)) * F_tank
T_m_tower = (L_tower / 2) * F_tower

print("Torsion moment for the entire Tower:\t{:.3f}  [N*m]".format(T_m_tank + T_m_tower))

print("\n--- %s seconds ---" % (time.time() - start_time))
