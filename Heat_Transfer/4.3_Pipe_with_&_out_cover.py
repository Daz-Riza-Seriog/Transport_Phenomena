# Code made for Sergio Andrés Díaz Ariza
# 12 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assessment 4.3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize

sns.set()

# Solve for Temperature of Steam at given Pressure
class enviroment_convective:

    def temp_steam_sat_reg(self, Sat_pressure_1, Sat_pressure_2, Temp_from_pressure_1, Temp_from_pressure_2,
                           Sat_pressure_system):
        p1 = Sat_pressure_1  # [kPa]
        p2 = Sat_pressure_2  # [kPa]
        T1 = Temp_from_pressure_1 + 273.15  # [K]
        T2 = Temp_from_pressure_2 + 273.15  # [K]
        P_x = Sat_pressure_system  # [kPa]
        m = (T2 - T1) / (p2 - p1)
        T = m * P_x - (m * p1) + T1
        return T

# Optimice for the maximum difference allow
class Optimice:
    def objective_T(self, x, *args):
        T_supp, r = args[0], args[1]
        thk = 0.015
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        return T_supp - ((x1 * np.log(r + thk)) - x2)

    def constraint_BC1_BC2(self, x):
        r, T_in = (0.025, 484.8362745098039)
        K, thk, h_in, T_out, h_out, e = (15.6, 0.015, 30, 25 + 273.15, 5, 0.3)
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        R_conv_1 = (1 / (2 * np.pi * (r)) * h_in)
        h_comb = (2 * np.pi * (r + thk)) * (h_out + e * 5.670e-8 * (x1 * np.log(r + thk) + x2 - T_out)
                                            * ((x1 * np.log(r + thk) + x2) ** 2 + T_out ** 2))
        R_cond = np.log(thk) / (2 * np.pi * K)
        return ((T_in - T_out) / (R_conv_1 + R_cond + (1 / h_comb))) + ((K * x1) / r)

    def objective_T_II(self, x, *args):
        T_supp, r = args[0], args[1]
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        return T_supp - ((x1 * np.log(r)) - x2)

    def constraint_BC1_BC2_II(self, x):
        r, T_in = (0.025, 484.8362745098039)
        K, thk_1, h_in, T_out, h_out = (15.6, 0.015, 30, 25 + 273.15, 5)
        K_2, thk_2, e = (0.25, 0.012, 0.8)
        x1 = x[0]  # C1
        x2 = x[1]  # C2
        R_conv_1 = (1 / (2 * np.pi * r) * h_in)
        R_cond = np.log(thk_1) / (2 * np.pi * K)
        R_cond_2 = np.log(thk_2) / (2 * np.pi * K_2)
        h_comb = (2 * np.pi * (r + thk_1 + thk_2)) * (
                h_out + e * 5.670e-8 * (x1 * np.log(r + thk_1 + thk_2) + x2 - T_out)
                * ((x1 * np.log(r + thk_1 + thk_2) + x2) ** 2 + T_out ** 2))
        return ((T_in - T_out) / (R_conv_1 + R_cond + R_cond_2 + (1 / h_comb))) + ((K * x1) / r)

# Determine the Q flux with cover and without cover
class Q_determine:

    def Q_uncover(self, r, T_in, K, thk, h_in, T_out, h_out, e, Delta_T):
        T_surf = (T_int - Delta_T) + 273.15
        R_conv_1 = (1 / (2 * np.pi * r) * h_in)
        h_comb = (2 * np.pi * (r + thk)) * (h_out + e * 5.670e-8 * (T_surf - T_out)
                                            * (T_surf ** 2 + T_out ** 2))
        R_cond = np.log(thk) / (2 * np.pi * K)
        Q = ((T_in - T_out) / (R_conv_1 + R_cond + (1 / h_comb)))
        return Q

    def Q_cover(self, r, T_in, K, K_2, thk_1, thk_2, h_in, T_out, h_out, e, Delta_T):
        T_surf = (T_int - Delta_T) + 273.15
        R_conv_1 = (1 / (2 * np.pi * r) * h_in)
        R_cond = np.log(thk_1) / (2 * np.pi * K)
        R_cond_2 = np.log(thk_2) / (2 * np.pi * K_2)
        h_comb = (2 * np.pi * (r + thk_1 + thk_2)) * (
                h_out + e * 5.670e-8 * (T_surf - T_out)
                * (T_surf ** 2 + T_out ** 2))
        Q = ((T_in - T_out) / (R_conv_1 + R_cond + R_cond_2 + (1 / h_comb)))
        return Q

# Temperature of T in of the cylinder iron
class T_profile_iron:

    def T_in_II(self, Q_tot, r, K, thk, T_surf_out):
        R_cond = np.log(r - thk) / (2 * np.pi * K)
        T_surf_in = (-Q_tot * R_cond) + T_surf_out
        return T_surf_in


env_conv = enviroment_convective()
Opt = Optimice()
Q_s = Q_determine()
T_iron = T_profile_iron()

T_int = env_conv.temp_steam_sat_reg(1553, 2318, 200, 220, 2000)

constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_BC1_BC2}

constraint = [constraint_equal1]
#           T_suppose, Radius_max, T_in
arguments = (T_int, 0.025)

x0 = [0, 0]  # This initial values are extracted from a first solution given by the method
sol = minimize(Opt.objective_T, x0, method='SLSQP', args=arguments, constraints=constraint, options={'maxiter': 5})

# BIG NOTE: modify the iteration to reach values according to reality--> You need more restrictions
# In the result you find the maximum difference that the system reach between the suppose and the reality

Q_1 = Q_s.Q_uncover(0.025, T_int, 15.6, 0.015, 30, 25 + 273.15, 5, 0.3, sol.fun)
T_in_iron = T_iron.T_in_II(Q_1, 0.025, 30, 0.015, (T_int - sol.fun) + 273.15)

########################################### Case 2 #####################################################################

constraint_equal1_II = {'type': 'eq', 'fun': Opt.constraint_BC1_BC2_II}

constraint_II = [constraint_equal1_II]
#           T_suppose, Radius_max
arguments_II = (T_int, 0.025 + 0.015 + 0.012)

x0 = [0, 0]  # This initial values are extracted from a first solution given by the method
sol_II = minimize(Opt.objective_T, x0, method='SLSQP', args=arguments_II, constraints=constraint_II,
                  options={'maxiter': 5})

# BIG NOTE: modify the iteration to reach values according to reality--> You need more restrictions
# In the result you find the maximum difference that the system reach between the suppose and the reality

Q_2 = Q_s.Q_cover(0.025, T_int, 15.6, 0.25, 0.015, 0.012, 30, 25 + 273.15, 5, 0.3, sol_II.fun)


print("========================= WITH UNCOVER ==============================================\n")
print("Temperature in the convective enviro. 1: {} [K]".format(T_int))
print("Temperature at the start of the cylinder: {} [K]".format(T_in_iron))
print("Temperature at the end of the cylinder: {} [K]".format((T_int - sol.fun) + 273.15))
print("Q for meter of cylinder: {} [W/m]\n".format(Q_1))
print("================================================================================")
print("========================= WITH COVER ==============================================\n")
print("Temperature in the convective enviro. 1: {} [K]".format(T_int))
print("Temperature at the end of the cylinder: {} [K]".format((T_int - sol_II.fun) + 273.15))
print("Q for meter of cylinder: {} [W/m]\n".format(Q_2))
print("================================================================================\n")