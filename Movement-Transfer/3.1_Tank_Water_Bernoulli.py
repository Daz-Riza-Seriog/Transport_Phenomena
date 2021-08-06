# Code made for Sergio Andrés Díaz Ariza
# 25 July 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 2.1

from scipy.integrate import quad
import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()

# Tank Parameters

Vol = 1.5  # [L] Capacity o the Tank
Dt = 1.2  # [m] Diameter of the tank
# Vol = A * z_t  --> Area * height of tank
z_t = (4 * Vol) / (np.pi * (Dt ** 2))

# Bernoulli Eq for the floors

g = 9.89  # [m/s^2] Value of Gravity acceleration
Patm = 101325  # [Pa]
P1 = P2 = P3 = P4 = Patm
z_4 = 9 + z_t  # Height of the tank
z_3 = 6  # Height of the third floor
z_2 = 3  # Height of the second floor
z_1 = 0  # Height of the first floor

V_4 = 0
V_3 = np.sqrt(2 * g * (z_4 - z_3))
V_2 = np.sqrt(2 * g * (z_4 - z_2))
V_1 = np.sqrt(2 * g * (z_4 - z_1))

# Find the Caudal Qi
Ds = 0.0254  # [m] Diameter of outside 1 inch

Q_1 = (np.pi * (Ds ** 2) * V_1) / 4
Q_2 = (np.pi * (Ds ** 2) * V_2) / 4
Q_3 = (np.pi * (Ds ** 2) * V_3) / 4
Q_4 = (np.pi * (Ds ** 2) * V_4) / 4

print(V_4, V_3, V_2, V_1)
print(Q_4, Q_3, Q_2, Q_1)

# Time to empty the tank to floor 1

t_floor_1 = (-2 / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * (np.sqrt(z_t - z_1) - np.sqrt(9 - z_1))

print(t_floor_1 / 60)


# Time to empty he tank to the three floors

# Define the Integral
def Tim_tank(x, z_1, z_2, z_3):
    return 1 / (np.sqrt(x - z_1) + np.sqrt(x - z_2) + np.sqrt(x - z_3))


# Solve the Integrate
sol_I = quad(Tim_tank, 9, z_t, args=(z_1, z_2, z_3))

t_floor_all = (1 / np.sqrt(2 * g)) * ((Dt / Ds) ** 2) * sol_I[0]

print(t_floor_all / 60)

print("\n--- %s seconds ---" % (time.time() - start_time))
