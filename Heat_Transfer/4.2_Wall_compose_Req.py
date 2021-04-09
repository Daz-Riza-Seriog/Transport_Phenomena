# Code made for Sergio Andrés Díaz Ariza
# 08 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assesment 3.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize

sns.set()


# Optimice the function for T, and assign constraints to resolve for Rmin,E_cons,C1,C2
class Optimice:
    def objective_Q(self, x):
        x1 = x[0]  # C1
        x2 = x[1]  # C2

        return 775 / ((1 / 65) + (0.5 / 0.2) + (
                (1 / 10 * 0.8 * 5.670e-8 * (x1 * 0.5 + x2 + 298.15) * ((x1 * 0.5 + x2) ** 2 + 298.15 ** 2))
                / ((1 / 10) + (
                0.8 * 5.670e-8 * (x1 * 0.5 + x2 + 298.15) * ((x1 * 0.5 + x2) ** 2 + 298.15 ** 2)))))

    def constraint_BC1_x_0_Q(self, x):
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        return 65 * (1073.15 - x2) + 0.2 * x1

    def constraint_BC2_x_L(self, x):
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        return (1073.15 - x2) + ((65 * 0.5 * (1073.15 - x2)) / 0.2) - (1073.15 - 0.5 * x1 - x2)

    def constraint_BC3_x_L_Q(self, x):
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        return 0.2 * x1 + 10 + (0.8 * 5.670e-8 * (x1 * 0.5 + x2 + 298.15) * ((x1 * 0.5 + x2) ** 2 + 298.15 ** 2)) \
               * (x1 * 0.5 + x2 - 298.15)

# Profile of Temperature of the Wall
class T_profile():
    def T(self, C1, C2, x):
        T = C1 * x + C2
        return T


Opt = Optimice()
constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_BC1_x_0_Q}
constraint_equal2 = {'type': 'eq', 'fun': Opt.constraint_BC2_x_L}
#constraint_equal3 = {'type': 'eq', 'fun': Opt.constraint_BC3_x_L_Q} --> Over fixed system

constraint = [constraint_equal1, constraint_equal2]
x0 = [-1500, 1073]  # This initial values are extracted from a first solution given by the method
sol = minimize(Opt.objective_Q, x0, method='SLSQP', constraints=constraint, options={'maxiter': 1000})

T_prof = T_profile()
length = np.arange(0, 0.5, 0.0001)
T_lenghth = T_prof.T(sol.x[0], sol.x[1], length)

plt.figure(1)
plt.plot(length, T_lenghth, 'c')
plt.title("Temperature Profile in a Wall", fontsize=16)
plt.ylabel("Temperature $[K]$", fontsize=14)
plt.xlabel("Long of pipe $[m]$ ", fontsize=14)

print("==========================================================")
print(f"Coefficient C1 of the Profile: {sol.x[0]}")
print(f"Coefficient C2 of the Profile: {sol.x[1]}")
print(f"Temperature of Wall in x=0: {T_lenghth[0]} [K]")
print(f"Temperature of Wall in x=0.5 : {T_lenghth[4999]} [K]")
print("==========================================================")
plt.show()
