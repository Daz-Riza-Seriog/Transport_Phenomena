# Code made for Sergio Andrés Díaz Ariza
# 04 August 2021
# License MIT
# Transport Phenomena: Python Quizz Point 2

from scipy.optimize import minimize
import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()

# Parameters  of Friction
eps = 1.5e-4
L1 = 20  # Length of pipe [m]
L2 = 6  # Length of pipe [m]

rho = 987 # Density of carbohydrates at 38 C  [Kg/m^3]
miu = 8.91e-4  # [Kg/m*s]


# Tank Parameters

Vol = 2.5  # [m^3] Capacity o the Tank
Dt = 1.2  # [m] Diameter of the tank
# Vol = A * z_t  --> Area * height of tank
z_t = (4 * Vol) / (np.pi * (Dt ** 2))

# Bernoulli Eq for the floors

g = 9.89  # [m/s^2] Value of Gravity acceleration
Patm = 101325  # [Pa]
P1 = P2 = P3 = P4 = Patm
z_4 = L1 + z_t  # Height of the tank
z_1 = 0  # Height of the third floor
L = L1 + L2
Ds = 0.0508

# Resolve the Balance of Mechanic Energy

class Optimice:
    def objective_Colebrook(self, x, *args):
        # Parameters
        # eps = 0
        # D = 0.0254  # Diameter
        # L = 300  # Length of pipe [m]
        # rough = 6e-4
        # rho = 680  # Density of carbohydrates at 38 C  [Kg/m^3]

        x1 = x[0]  # Darcy factor
        x2 = x[1]  # Velocity Average
        x3 = x[2]  # Head Loss

        return (1 / np.sqrt(x1)) + (2.0 * np.log10(((eps / Ds) / 3.7) + (2.51 / (((rho * Ds * x2) / miu) * np.sqrt(x1)))))

    def constraint_hl_eq_f(self, x, *args):
        # Parameters

        x1 = x[0]  # Darcy factor
        x2 = x[1]  # Velocity Average
        x3 = x[2]  # Head Loss


        return x3 - (x1 * (L * (x2 ** 2) / Ds * 2 * g))

    def constraint_Vavg_eq_hl(self, x, *args):
        # Parameters
        x1 = x[0]  # Darcy factor
        x2 = x[1]  # Velocity Average
        x3 = x[2]  # Head Loss


        return x2 - np.sqrt(2 * g * (z_4 - z_1) + x3)





Opt = Optimice()
constraint_equal = {'type': 'eq', 'fun': Opt.objective_Colebrook}
constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_hl_eq_f}
constraint_equal2 = {'type': 'eq', 'fun': Opt.constraint_Vavg_eq_hl}

constraint = [constraint_equal, constraint_equal1, constraint_equal2]

x0 = [0.05,0.05,0.001]  # This inital values are extracted from a first solution given by the method
sol = minimize(Opt.objective_Colebrook, x0, method='SLSQP', constraints=constraint, args=(eps, rho, miu, L, g, z_4, z_1, Ds),
               options={'maxiter': 1000})

print(sol)


print("\n--- %s seconds ---" % (time.time() - start_time))