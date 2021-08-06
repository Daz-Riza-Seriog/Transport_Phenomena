# Code made for Sergio Andrés Díaz Ariza
# 29 July 2021
# License MIT
# Transport Phenomena: Pipe find Caudal

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
        eps = 0
        D = 0.0762  # Diameter
        L = 300  # Length of pipe [m]
        rough = 6e-4
        rho = 680  # Density of carbohydrates at 38 C  [Kg/m^3]
        niu = 3.62e-7  # Cinematic Viscosity [m^2/s]
        miu = niu * rho
        dP = 172000  # Pressure Drop [Pas]

        x1 = x[0]  # Darcy factor

        V_avg = np.sqrt(2 * D * x1 * dP / rho * L)  # Average Velocity [m/s]
        Re = (rho * D * V_avg) / miu  # Reynolds Number

        return 1 / np.sqrt(x1) + 2.0 * np.log10((rough / 3.7) + (2.51 / (Re * np.sqrt(x1))))

class Vel_average:
    def Velocity(self, dP, f, D, rho, L):
        # dP = Drop Pressure [Pas]
        # f = Darcy factor
        # rho = density of carbohydrates [Kg/m^3]
        # L = length of the pipe [m]
        # V =  Average Flow [m^3/seg]
        # D = Diameter of pipe [m]
        return np.sqrt(2 * D * f * dP / rho * L)  # Average Velocity [m/s]

    def V(self, Vel, D):
        # Vel =  Average Velocity [m/seg]
        # D = Diameter of pipe [m]
        return np.pi*(D**2)/4  # Average Velocity [m/s]


Opt = Optimice()
constraint_equal = {'type': 'eq', 'fun': Opt.objective_Colebrook}
constraint = [constraint_equal]

x0 = [0.05]  # This inital values are extracted from a first solution given by the method
sol = minimize(Opt.objective_Colebrook, x0, method='SLSQP',constraints=constraint, options={'maxiter': 1000})

Vel_av = Vel_average()
Vel = Vel_av.Velocity(172000, sol.x[0], 0.0762, 680, 300)
Flow = Vel_av.V(Vel,0.0254)

print(sol)
print("\nDarcy factor :\t", sol.x[0])
print("\nAverage Velocity:\t", Vel, "m/s")
print("\nAverage Caudal:\t", Flow, "m^3/s")

print("\n--- %s seconds ---" % (time.time() - start_time))
