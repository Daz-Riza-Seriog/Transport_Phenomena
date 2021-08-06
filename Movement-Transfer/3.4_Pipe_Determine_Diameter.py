# Code made for Sergio Andrés Díaz Ariza
# 29 July 2021
# License MIT
# Transport Phenomena: Pipe find Diameter

from scipy.optimize import minimize
import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()


# Optimice the function for T, and assign constraints to resolve for Rmin,E_cons,C1,C2
class Optimice:
    def objective_Colebrook(self, x):
        # Parameters
        eps = 2.6e-4  # Roughness [m]
        L = 1200  # Length of pipe [m]
        niu = 1.3e-7  # Cinematic Viscosity [m^2/s]
        DP = 2  # Head Drop [m]
        V = 0.55  # Caudal [m^3/s]

        x1 = x[0]  # Darcy factor
        x2 = x[1]  # Diameter
        x3 = x[2]  # Velocity Average

        return (1 / np.sqrt(x1)) + (2.0 * np.log10(
            ((eps / (x1 * L * (x3 ** 2) / DP * 2)) / 3.7) + (2.51 / ((V * x2 / niu) * np.sqrt(x1)))))

    def constraint_D_eq_f(self, x):
        # Parameters
        L = 1200  # Length of pipe [m]
        DP = 2  # Head Drop [m]

        x1 = x[0]  # Darcy factor
        x2 = x[1]  # Diameter
        x3 = x[2]  # Velocity Average
        return x2 - (x1 * (L * (x3 ** 2) / DP * 2))

    def constraint_Vavg_eq_D(self, x):
        # Parameters
        V = 0.55  # Caudal [m^3/s]

        x2 = x[1]  # Diameter
        x3 = x[2]  # Velocity Average
        return x3 - (4 * V / (np.pi * (x2 ** 2)))


Opt = Optimice()
constraint_equal = {'type': 'eq', 'fun': Opt.objective_Colebrook}
constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_D_eq_f}
constraint_equal2 = {'type': 'eq', 'fun': Opt.constraint_Vavg_eq_D}

constraint = [constraint_equal, constraint_equal1, constraint_equal2]

x0 = [0.5, 1, 1.5]
sol = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, options={'maxiter': 1000})

print(sol)
print("\nDarcy factor :\t", sol.x[0])
print("\nDiameter:\t", sol.x[1], "[m]")
print("\nVelocity Average:\t", sol.x[2], "[m/s]")

print("\n--- %s seconds ---" % (time.time() - start_time))
