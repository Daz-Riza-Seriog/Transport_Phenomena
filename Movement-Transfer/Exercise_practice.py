# Code made for Sergio Andrés Díaz Ariza
# 29 July 2021
# License MIT
# Transport Phenomena: Pipe with  Contraction, find Hloss

from scipy.optimize import minimize
import seaborn as sns
import numpy as np
import time

start_time = time.time()
sns.set()


# Optimice the function for T, and assign constraints to resolve for Rmin,E_cons,C1,C2
class Optimice:
    def objective_Colebrook_1(self, x):
        # Parameters
        eps = 0
        D = 0.05  # Diameter
        rho = 997  # Density of water at 25 C  [Kg/m^3]
        V = 7e-3  # Average Flow [m^3/seg]
        V_avg = V / ((np.pi * D ** 2) / 4)  # Average Velocity [m/s]
        miu = 8.91e-4
        Re = (rho * D * V_avg) / miu

        x1 = x[0]  # Darcy factor

        return 1 / np.sqrt(x1) + 2.0 * np.log10(((eps / D) / 3.7) + (2.51 / (Re * np.sqrt(x1))))

    def objective_Colebrook_2(self, x):
        # Parameters
        eps = 0
        D = 0.12  # Diameter
        rho = 997  # Density of water at 25 C  [Kg/m^3]
        V = 5.666e-3  # Average Flow [m^3/seg]
        V_avg = V / ((np.pi * D ** 2) / 4)  # Average Velocity [m/s]
        miu = 8.91e-4
        Re = (rho * D * V_avg) / miu

        x1 = x[0]  # Darcy factor

        return 1 / np.sqrt(x1) + 2.0 * np.log10(((eps / D) / 3.7) + (2.51 / (Re * np.sqrt(x1))))

class Paramters:
    def Velocity(self, V, D):
        # V =  Average Flow [m^3/seg]
        # D = Diameter of pipe [m]
        return V / ((np.pi * D ** 2) / 4)  # Average Velocity [m/s]

class Head_Loss_Major:
    def h_l(self, f, L, D, V_avg, g):
        #   Parameters
        # f = Darcy factor
        # L = length of the pipe
        # D = Diameter of the pipe
        # V_avg = Velocity average of the fluid
        # g = constant of gravity

        return f * (L / D) * ((V_avg ** 2) / (2 * g))

class Head_Loss_Minor:
    def h_l(self, K_l, V_avg, g):
        #   Parameters
        # K_l = Loss coefficient
        # V_avg = Velocity average of the fluid
        # g = constant of gravity

        return K_l * ((V_avg ** 2) / (2 * g))

Opt = Optimice()
constraint_equal1 = {'type': 'eq', 'fun': Opt.objective_Colebrook_1}
constraint_equal2 = {'type': 'eq', 'fun': Opt.objective_Colebrook_2}

constraint = [constraint_equal1]
constraint2 = [constraint_equal2]
x0 = [0.05]  # This inital values are extracted from a first solution given by the method
sol1 = minimize(Opt.objective_Colebrook_1, x0, method='SLSQP', constraints=constraint, options={'maxiter': 1000})
sol2 = minimize(Opt.objective_Colebrook_2, x0, method='SLSQP', constraints=constraint2, options={'maxiter': 1000})

Params = Paramters()
V_avg_1 = Params.Velocity(7e-3, 0.05)

H_loss_M = Head_Loss_Major()
h_l_1_M = H_loss_M.h_l(sol1.x[0], 30, 0.05, V_avg_1, 9.82)

H_loss_m = Head_Loss_Minor()
h_l_1_m = H_loss_m.h_l(0.12, V_avg_1, 9.82)


print(sol1)
print(sol2)
print("\nDarcy factor for first part:\t", sol1.x[0])
print("\nDarcy factor for second part:\t", sol2.x[0])
print("\nHead loss Major for first part:\t", h_l_1_M)

print("\n--- %s seconds ---" % (time.time() - start_time))
