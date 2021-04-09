# Code made for Sergio Andrés Díaz Ariza
# 07 April 2021
# License MIT
# Transport Phenomena: Python Program-Assesment 4.2

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()

# Parameter m of the Tin
class Cylind_Param_Tin:

    def Param_m(self, h_convection, Kcylinder, Diameter_cm, Length_cyl_1_2):
        m = np.sqrt((4 * h_convection * Length_cyl_1_2) / (Kcylinder * Diameter_cm ))
        return m

# Coefficients to put in Vector B
class Coeff_vector_B:

    def Eq_1(self, T_final_wall_1):
        b1 = T_final_wall_1
        return b1

    def Eq_2(self):
        b2 = 0
        return b2

    def Eq_3(self):
        b3 = 0
        return b3

    def Eq_4(self, T_init_wall_2):
        b4 = T_init_wall_2
        return b4

# Solve he prolfes of Temperature
class T_profile:

    def T_x_to_Tf(self, C1_1, C2_1, x):
        T = C1_1 * x + C2_1
        return T

    def T_x_to_Tw(self, C1_2, C2_2, Param_m, x2):
        T = (C1_2 * np.exp(Param_m * x2)) + (C2_2 * np.exp(-Param_m * x2))
        return T


Cyl = Cylind_Param_Tin()
Vect_B = Coeff_vector_B()
Profile = T_profile()

Diam = 2.2  # [cm] Diameter of Cylinde
K = 60  # [W/m*k] Conductive coefficient
h = 15  # [W/m^2*k] Conductive coefficient
As = np.pi * (((Diam / 100) ** 2) / 4)  # [m^2] Cross Sectional Area
Rc = 4e-3  # [m^2*K/W] Contact Resistance

Param_m = Cyl.Param_m(h, K, Diam/100, 0.20)  # Param m for Tin

b1 = Vect_B.Eq_1(300 + 273.15)
b2 = Vect_B.Eq_2()
b3 = Vect_B.Eq_3()
b4 = Vect_B.Eq_4(50 + 273.15)

Matx1 = pd.DataFrame()
Matx1['B'] = pd.Series([b1, b2, b3, b4], index=['Eq1', 'Eq2', 'Eq3', 'Eq4'])
Matx2 = pd.DataFrame([[(-K * As * Rc) - 0.2, -1, 0, 0], [0, 1, -1, -1], [1, 0, -Param_m, Param_m],
                      [0, 0, (-Rc * K * As * Param_m * np.exp(Param_m * 0.2)) - np.exp(Param_m * 0.2),
                       (Rc * K * As * Param_m * np.exp(-Param_m * 0.2)) - np.exp(-Param_m * 0.2)]]
                     , columns=['1', '2', '3', '4'], index=['Eq1', 'Eq2', 'Eq3', 'Eq4'])
Matx = pd.concat([Matx2, Matx1], axis=1)

Matx_sol = pd.DataFrame(np.linalg.solve(Matx.iloc[:, 0:4], Matx.loc[:, ['B']]), columns=['Solution'],
                        index=['C1_I', 'C2_I', 'C1_II', 'C2_II'])  # Create a Vector solution

# solve for Temperature Profile
T_x_tf = Profile.T_x_to_Tf(Matx_sol.iloc[0, 0], Matx_sol.iloc[1, 0], np.arange(-0.2, 0, 1e-3))
T_x_Tw = Profile.T_x_to_Tw(Matx_sol.iloc[2, 0], Matx_sol.iloc[3, 0], Param_m, np.arange(0, 0.2, 1e-3))

print(Matx_sol)
print(Matx)

plt.figure(1)
plt.plot(np.arange(-0.2, 0, 1e-3), T_x_tf, label='$0<L_c<0.2$')
plt.plot(np.arange(0, 0.2, 1e-3), T_x_Tw, label='$0.2<L_c<0.4$')
plt.title("Temperature Profile in a Cylinder\n with Insulated part", fontsize=16)
plt.ylabel("Temperature $[K]$", fontsize=14)
plt.xlabel("Long of wire $[m]$ ", fontsize=14)
plt.legend()
plt.show()
