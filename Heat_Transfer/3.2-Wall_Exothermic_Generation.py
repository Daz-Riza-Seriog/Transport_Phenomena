# Code made for Sergio Andrés Díaz Ariza
# 21 March 2021
# License MIT
# Transport Phenomena: Python Program-Assesment 3.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
sns.set()

# Optimice the function for T, and assign constraints to resolve for Rmin,E_cons,C1,C2
class Optimice:
    def objective_T(self,x):
        x1 = x[0]   # E_gen
        x2 = x[1]   # C1
        x3 = x[2]   # C2
        x4 = x[3]   # x
        return ((-x1)*(x4**2)/4*0.8) + x2*x4 + x3

    def constraint_equal_Tmax_L_middle(self,x):
        x1 = x[0]  # E_gen
        x2 = x[1]  # C1
        x3 = x[2]  # C2
        return ((-x1)*((0.3/2)**2)/4*0.8) + x2*(0.3/2) + x3 - 523.15

    def constraint_equal_B_C_1_and_BC2(self,x):
        x1 = x[0]  # E_gen
        x2 = x[1]  # C1
        x4 = x[3]  # x
        return x1*x4*((15/30)+1)+15*353.15-15*298.15-x2*0.8*(1-(15/30))

    def constraint_equal_L_middle_Egen(self,x):
        x1 = x[0]  # E_gen
        x2 = x[1]  # C1
        return x1*(0.15/0.8)+x2


class T_profile():
    def T(self,x,E_gen,C1,C2,K):
        T = (-E_gen/2*K)*x**2 +C1*x + C2
        return T

Opt = Optimice()
constraint_equal1 = {'type':'eq','fun':Opt.constraint_equal_Tmax_L_middle}
constraint_equal2 = {'type':'eq','fun':Opt.constraint_equal_B_C_1_and_BC2}
constraint_equal3 = {'type':'eq','fun':Opt.constraint_equal_L_middle_Egen}

constraint = [constraint_equal1,constraint_equal2,constraint_equal3]
x0 = [0.1,0.1,0.1,0.15] # This inital values are extracted from a first solution given by the method
sol = minimize(Opt.objective_T,x0,method='SLSQP',constraints=constraint,options={'maxiter':1000})

T_prof = T_profile()
length = np.arange(0,0.3,0.0001)
T_length = T_prof.T(length,-sol.x[0],sol.x[1],sol.x[2],0.8)

plt.figure(1)
plt.plot(length,T_length,linestyle=(0, (3,1,1, 1)))
plt.title("Temperature Profile in a Wall\n with Exothermic Reaction ",fontsize= 16)
plt.ylabel("Temperature $[K]$",fontsize= 14)
plt.xlabel("Long of wall $[m]$ ",fontsize= 14)

print(sol)
plt.show()