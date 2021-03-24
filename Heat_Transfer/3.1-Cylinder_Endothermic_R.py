# Code made for Sergio Andrés Díaz Ariza
# 21 March 2021
# License MIT
# Transport Phenomena: Python Program-Assesment 3.1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
sns.set()

# Optimice the function for T, and assign constraints to resolve for Rmin,E_cons,C1,C2
class Optimice:
    def objective_T(self,x):
        x1 = x[0]   # R_min
        x2 = x[1]   # E_cons
        x3 = x[2]   # C1
        x4 = x[3]   # C2
        return (-(x1**2)*x2/4*12) + x3*np.log(x1) + x4

    def constraint_equal_Rmin(self,x):
        x1 = x[0]  # R_min
        x2 = x[1]  # E_cons
        x3 = x[2]  # C1
        return (-(x1**2)*x2/2*12) + x3

    def constraint_inequal_Tmin(self,x):
        x1 = x[0]  # R_min
        x2 = x[1]  # E_cons
        x3 = x[2]  # C1
        x4 = x[3]  # C2
        return (-(x1**2)*x2/4*12) + x3*np.log(x1) + x4 -275.15

    def constraint_inequal_Tsup_in(self,x):
        x2 = x[1]  # E_cons
        x3 = x[2]  # C1
        x4 = x[3]  # C2
        return (-(0.1**2)*x2/4*12) + x3*np.log(0.1) + x4 -283.15

    def constraint_inequal_Tsup_out(self,x):
        x2 = x[1]  # E_cons
        x3 = x[2]  # C1
        x4 = x[3]  # C2
        return (-(0.2**2)*x2/4*12) + x3*np.log(0.2) + x4 -288.15

    def constraint_equal_B_C_1(self,x):
        x2 = x[1]  # E_cons
        x3 = x[2]  # C1
        x4 = x[3]  # C2
        return 125*(323.15+(((0.1**2)*x2)/4*12)+x3*np.log(0.1)+x4)+12*((-0.1*x2/2*12)+x3/0.1)

class T_profile():
    def T(self,R,E_con,C1,K,C2):
        T = (-(R**2)*E_con/4*K)+C1*np.log(R) + C2
        return T

Opt = Optimice()
constraint_equal1 = {'type':'eq','fun':Opt.constraint_equal_Rmin}
constraint_equal2 = {'type':'ineq','fun':Opt.constraint_inequal_Tmin}
constraint_equal3 = {'type':'eq','fun':Opt.constraint_equal_B_C_1}
constraint_equal4 = {'type':'ineq','fun':Opt.constraint_inequal_Tsup_in}
constraint_equal5 = {'type':'ineq','fun':Opt.constraint_inequal_Tsup_out}
constraint = [constraint_equal1,constraint_equal2,constraint_equal3,constraint_equal4,constraint_equal5]
x0 = [0.15,-7e3,-6e2,-1e3] # This inital values are extracted from a first solution given by the method
sol = minimize(Opt.objective_T,x0,method='SLSQP',constraints=constraint,options={'maxiter':1000})

T_prof = T_profile()
radius = np.arange(0.1,0.2,0.0001)
T_radius = T_prof.T(radius,sol.x[1],sol.x[2],12,sol.x[3])

plt.figure(1)
plt.plot(radius,T_radius,linestyle=(0, (3,1,1, 1)))
plt.xlim(0.05,0.25)
plt.title("Temperature Profile in a Pipe\n with Endothermic Reaction ",fontsize= 16)
plt.ylabel("Temperatura $[K]$",fontsize= 14)
plt.xlabel("Long of pipe $[m]$ ",fontsize= 14)

print(sol)
plt.show()