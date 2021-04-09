# Code made for Sergio Andrés Díaz Ariza
# 03 April 2021
# License MIT
# Transport Phenomena: Python Program-Case of Study 2-Wire-Cover

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from sympy import symbols, Eq, solve
from scipy.optimize import minimize

sns.set()


# Analysis in constant Generation of Heat for electric Current
class Copper_Wire:

    def BC1_Egen_eq_Q(self, Q_gen, Radius_copper, length_wire_copper):
        Q = Q_gen  # [W]
        r = Radius_copper  # [m]
        L_c = length_wire_copper  # [m]
        E_gen = Q * 2 / (2 * np.pi * r * L_c) * r  # [W/m^3]
        return E_gen

    def BC2_Q_eq_Therm_Cont_Resist(self, h_contact_copper, Apparent_Area, T_surface_copper_out, T_suface_PVC_in):
        h_cc = h_contact_copper  # [W/m^2*K]
        App_A = Apparent_Area  # [m^2]
        T_sc = T_surface_copper_out  # [K]
        T_sPVC_in = T_suface_PVC_in  # [K]
        Q = h_cc * App_A * (T_sc - T_sPVC_in)  # [W]
        return Q

    def C2_coefficient(self, E_gen, K_copper, T_surface_copper_out, Radius_copper):
        E_g = E_gen  # [W/m^3]
        r = Radius_copper  # [m]
        K_c = K_copper  # [W/m*K]
        T_sc = T_surface_copper_out  # [K]
        C2 = T_sc + (E_g * (r ** 2)) / 4 * K_c
        return C2


# Using Ohm's Law and Power for determine Voltage and Electric Current
class Voltage_Current:
    def Voltage(self, Reistance_copper_wire, Power_gen):
        R_c = Reistance_copper_wire  # [ohm/m]
        P_gn = Power_gen  # [W]
        V = np.sqrt(R_c * P_gn)  # [Volt]
        return V

    def Current(self, Reistance_copper_wire, Power_gen):
        R_c = Reistance_copper_wire  # [ohm/m]
        P_gn = Power_gen  # [W]
        I = np.sqrt(P_gn / R_c)  # [A]
        return I


# Determine the Temperature in PVC cover
class PVC_1_cover:

    def B_C_1_r_in_Q_C1(self, E_gen, Radius_min_PVC, K_pvc_1):
        E = E_gen  # [W/m^3]
        K_pvc = K_pvc_1  # [W/m*K]
        r = Radius_min_PVC  # [m]
        C1 = E * (r ** 2) / (-K_pvc * 2)  # [K]
        return C1

    def B_C_2_r_max_Resis(self, E_gen, Radius_min_pvc_1, Radius_max_pvc_1, T_surface_in_pvc_1, K_conductiviy_pvc,
                          C1_pvc_1, length_wire):
        E = E_gen  # [W/m^3]
        r_min = Radius_min_pvc_1  # [m]
        r_max = Radius_max_pvc_1  # [m]
        l_w = length_wire  # [m]
        T_s = T_surface_in_pvc_1  # [K]
        K_c = K_conductiviy_pvc  # [W/m*K]
        C1 = C1_pvc_1  # [K]
        As = np.pi * r_max * l_w
        L_r = r_max - r_min  # [m]
        Rc = L_r / (K_c * As)  # [W/m]
        Q = E * ((np.pi * (r_max ** 2) * l_w) - (np.pi * (r_min ** 2) * l_w))  # [W]
        C2 = T_s + (-Q * Rc) - (C1 * np.log(r_max))  # [K]
        return C2


# Using sympy library for find roots of Equation Heat in PVC_1 ext and T surrounding
class Convective_Zone:

    def T_surr_in(self, C1_pvc_1, K_conductivity_pvc, hconv_air, emiss_pvc, T_surface_out_pvc_1, Radius_max_PVC_1):
        K_c = K_conductivity_pvc  # [W/m*K]
        h_c = hconv_air  # [W/m^2*K]
        em_pvc = emiss_pvc  # []
        boltz = 5.670e-8  # [W/m^2*K^4]
        T_s = T_surface_out_pvc_1  # [K]
        r_max = Radius_max_PVC_1  # [m]
        C1 = C1_pvc_1  # [K]
        x = symbols('x')
        eq1 = Eq((-K_c * C1 / r_max) - (h_c + em_pvc * boltz * (T_s + x) * ((T_s ** 2) + (x ** 2))) * (T_s - x), 0)
        sol = solve(eq1, x)
        return sol


# Determine the Profile of PVC_2
class PVC_2_cover:

    def B_C_1_Q_conv_eq_cond(self, hconv_air, emiss_pvc, T_surface_out_pvc_1, T_surr_in, Radius_max_PVC_1
                             , Radius_min_pvc_2, K_conductivity_pvc):
        K_c = K_conductivity_pvc  # [W/m*K]
        h_c = hconv_air  # [W/m^2*K]
        em_pvc = emiss_pvc  # []
        boltz = 5.670e-8  # [W/m^2*K^4]
        T_s = T_surface_out_pvc_1  # [K]
        T_surr = T_surr_in  # [K]
        r_max = Radius_max_PVC_1  # [m]
        r_min_pvc2 = Radius_min_pvc_2  # [m]
        h_comb = (h_c + em_pvc * boltz * (T_s + T_surr) * ((T_s ** 2) + (T_surr_in ** 2))) * (T_s - T_surr)  # [k]
        C1 = r_max * h_comb * r_min_pvc2 / -K_c * r_min_pvc2
        return C1

    def B_C_2_T_suf_ext(self, T_surface_ext_PVC, C1_PVC_2, R_max_PVC_2):
        T_ext = T_surface_ext_PVC  # [K]
        C1 = C1_PVC_2  # [K]
        r_max = R_max_PVC_2  # [m]
        C2 = T_ext - C1 * np.log(r_max)  # [K]
        return C2


# minimize Temperature for copper without cover
class Optimice_Copper_Alone:
    def objective_T(self, x):
        x1 = x[0]  # E_gen [W/m^3]
        x2 = x[1]  # C1 [K]
        x3 = x[2]  # C2 [K]
        return (-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log((2.32 / 2)) + x3

    def constraint_equal_Q_Egen(self, x):
        x1 = x[0]  # E_gen [W/m^3]
        x4 = x[3]  # Q [W]
        return x4 + np.pi * ((2.32 / 2) ** 2) * 25 * x1

    def constraint_equal_Q_Resist(self, x):
        x1 = x[0]  # E_gen [W/m^3]
        x2 = x[1]  # C1 [K]
        x3 = x[2]  # C2 [K]
        x4 = x[3]  # Q [W]
        return (x4 * ((1 / 2.5 * np.pi * 2.32 * 25) * (1 / 0.74 * 5.670e-8 * (
                    (-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log((2.32 / 2)) + x3 - 217 + 273.15)
                                                       * (((-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log(
                    (2.32 / 2)) + x3) ** 2 - (217 + 273.15) ** 2)))
                / ((1 / 2.5 * np.pi * 2.32 * 25) + (1 / 0.74 * 5.670e-8 * (
                            (-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log((2.32 / 2)) + x3 - 217 + 273.15)) *
                   (((-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log((2.32 / 2)) + x3) ** 2 - (217 + 273.15) ** 2))) - \
               ((-x1 * ((2.32 / 2) ** 2) / 4 * 386) + x2 * np.log((2.32 / 2)) + x3) + (217 + 273.15)

    def Q(self, h_conv_air, Emissivity_copper, Temp_surface_copper, Temp_max_PVC_2):
        h_conv = h_conv_air  # [W/m^2*K]
        emiss = Emissivity_copper  # []
        boltz = 5.670e-8  # [W/m^2*K^4]
        T1 = Temp_surface_copper  # [K]
        T2 = Temp_max_PVC_2  # [K]
        T = T1 - T2  # [K]
        Rcomb = ((1 / h_conv * np.pi * 2.32 * 25) * (1 / emiss * boltz * (T1 - T2) * ((T1 ** 2) - (T2 ** 2))) \
                 / ((1 / h_conv * np.pi * 2.32 * 25) + (1 / emiss * boltz * (T1 - T2) * ((T1 ** 2) - (T2 ** 2)))))
        Q = T / Rcomb  # [@]
        return Q


# Profiles of Temperature of each layer
class T_profiles:

    def Prof_copper(self, E_gen, K_copper, Radius, C2):
        E_g = E_gen  # [W/m^3]
        r = Radius  # [m]
        K_c = K_copper  # [W/m*K]
        T = ((-E_g) * (r ** 2)) / (4 * K_c) + C2
        return T

    def Prof_PVC_1(self, C1_PVC_1, C2_PVC_1, Radius):
        r = Radius  # [m]
        C1 = C1_PVC_1  # [K]
        C2 = C2_PVC_1  # [K]
        T = C1 * np.log(r) + C2
        return T

    def Prof_PVC_2(self, C1_PVC_2, C2_PVC_2, Radius):
        r = Radius  # [m]
        C1 = C1_PVC_2  # [K]
        C2 = C2_PVC_2  # [K]
        T = C1 * np.log(r) + C2
        return T


# Solve for First part of the System
Cop_Wir = Copper_Wire()
T_gap = 90  # Maximum temperature of work wire 12 AWG-literature in radius maximum
Q_gen = Cop_Wir.BC2_Q_eq_Therm_Cont_Resist(55000, ((2.32 / 1000) / 2) * 25 * 2 * np.pi, T_gap + 273.15,
                                           75 + 273.15)  # [W]--Power
E_gen = Cop_Wir.BC1_Egen_eq_Q(Q_gen, ((2.32 / 1000) / 2), 25)  # [W/m^3]--Energy Generated
C2_copp = Cop_Wir.C2_coefficient(E_gen, 386, T_gap + 273.15, (2.32 / 1000) / 2)

# Solve for Voltage and Current
Volt_Curr = Voltage_Current()
Voltage = Volt_Curr.Voltage(5.32 - 3, Q_gen)
Current = Volt_Curr.Current(5.32 - 3, Q_gen)

# Solve for PVC 1
PVC_1 = PVC_1_cover()
C1_pvc_1 = PVC_1.B_C_1_r_in_Q_C1(E_gen, (2.32 / 1000) / 2, 0.17)
C2_pvc_1 = PVC_1.B_C_2_r_max_Resis(E_gen, ((2.32 / 1000) / 2), (((2.32 / 2) + 0.76) / 1000), 75 + 273.15, 0.17,
                                   C1_pvc_1, 25)

# Arrays for Plotting and Solve for T's Copper and PVC_1
T_prof = T_profiles()
R_copp = np.arange(0, (2.32 / 1000) / 2, 1e-6)
R_pvc_1 = np.arange(((2.32 / 2) / 1000), (((2.32 / 2) + (0.76)) / 1000), 1e-6)

# Profile Arrays for T copper and PVC 1
Prof_copper = T_prof.Prof_copper(E_gen, 386, R_copp, C2_copp)
Prof_pvc_1 = T_prof.Prof_PVC_1(C1_pvc_1, C2_pvc_1, R_pvc_1)

# Solve for Temperature in convective Zone in
Conv = Convective_Zone
T_surr_in = Conv.T_surr_in(0, C1_pvc_1, 0.17, 25, 0.92, Prof_pvc_1[759], (((2.32 / 2) + 0.76) / 1000))

# Solve for PVC2
PVC_2 = PVC_2_cover()
C1_pvc_2 = PVC_2.B_C_1_Q_conv_eq_cond(25, 0.92, Prof_pvc_1[759], T_surr_in[1], (((2.32 / 2) + 0.76) / 1000),
                                      7.285 / 1000, 0.17)
C2_pvc_2 = PVC_2.B_C_2_T_suf_ext(73 + 273.15, C1_pvc_2, 10.565 / 1000)

# Arrays for Plotting and Solve for T's PVC_2
R_pvc_2 = np.arange(7.285 / 1000, 10.565 / 1000, 1e-6)

# Profile Arrays for T copper and PVC 1
Prof_pvc_2 = T_prof.Prof_PVC_2(C1_pvc_2, C2_pvc_2, R_pvc_2)

####################################################
#   Second Part Study Case Copper without recover  #
####################################################

# Optimice for Temperature of Copper / Many variables few B.C
Opt = Optimice_Copper_Alone()
constraint_equal1 = {'type': 'eq', 'fun': Opt.constraint_equal_Q_Egen}
constraint_equal2 = {'type': 'eq', 'fun': Opt.constraint_equal_Q_Resist}

constraint = [constraint_equal1, constraint_equal2]
x0 = [1.28212852e-02, 7.04142154e+02, 3.87306067e+02,
      -1.35499415e+00]  # This inital values are extracted from a first solution given by the method
sol = minimize(Opt.objective_T, x0, constraints=constraint, method='SLSQP', options={'maxiter': 1000})

# Solve for Parameter of Copper without cover
Q_copper_without_cov = Opt.Q(25, 0.74, sol.fun, (217 + 273.15))
E_gen_coper_2 = Cop_Wir.BC1_Egen_eq_Q(Q_copper_without_cov, ((2.32 / 1000) / 2), 25)  # [W/m^3]--Energy Generated
C2_copp_2 = Cop_Wir.C2_coefficient(E_gen_coper_2, 386, sol.fun, (2.32 / 1000) / 2)
Voltage2 = Volt_Curr.Voltage(5.32 - 3, Q_copper_without_cov)
Current2 = Volt_Curr.Current(5.32 - 3, Q_copper_without_cov)

# Profile Arrays for T copper without
Prof_copper2 = T_prof.Prof_copper(E_gen_coper_2, 386, R_copp, C2_copp_2)
mod_T = sol.fun - (Prof_copper2[0] / 1e3)

# Print parameters
print("============================================================")
print(f"Power generated for the system wire:\t{Q_gen} [W]")
print(f"Energy generated for the system wire:\t{E_gen} [W/m^3]")
# print(f"C2 coefficient for the system wire:\t{C2_copp} [K]")
print(f"Voltage through the system wire:\t{Voltage} [Vol]")
print(f"Current Through the system wire:\t{Current} [A]")
print("============================================================")
print(f"Power generated for the system wire:\t{Q_copper_without_cov / 1e3} [W]")
print(f"Energy generated for the system wire:\t{E_gen_coper_2 / 1e3} [W/m^3]")
# print(f"C2 coefficient for the system wire:\t{C2_copp_2} [K]")
print(f"Voltage through the system wire:\t{Voltage2 / 1e2} [Vol]")
print(f"Current Through the system wire:\t{Current2 / 1e2} [A]")

# Plotting
plt.figure(1)
gspec = gridspec.GridSpec(2, 4)
top_plot = plt.subplot(gspec[0, 0:4])
lef_1_plot = plt.subplot(gspec[1, 0])
lef_2_plot = plt.subplot(gspec[1, 1])
lef_3_plot = plt.subplot(gspec[1, 2])
lef_4_plot = plt.subplot(gspec[1, 3])
top_plot.plot(R_copp * 1000, Prof_copper, label='$T_{copper}$ $[K]$')
top_plot.vlines(x=(2.32 / 2), ymin=75 + 273.15, ymax=Prof_copper[1159], label='$T_{gap}$ $[K]$', color='r')
top_plot.plot(R_pvc_1 * 1000, Prof_pvc_1, label='$T_{pvc_{1}}$ $[K]$', color='g')
top_plot.plot(R_pvc_2 * 1000, Prof_pvc_2, label='$T_{pvc_{2}}$ $[K]$', color='m')
lef_1_plot.plot(R_copp * 1000, Prof_copper)
lef_2_plot.vlines(x=(2.32 / 2), ymin=75 + 273.15, ymax=Prof_copper[1159], label='$T_gap_$', colors='r')
lef_3_plot.plot(R_pvc_1 * 1000, Prof_pvc_1, 'g')
lef_4_plot.plot(R_pvc_2 * 1000, Prof_pvc_2, 'm')
top_plot.legend()

plt.figure(2)
plt.plot(R_copp * 1000, Prof_copper - 273.15)
plt.title("Perfil de Temperatura vs Radio del Cobre", fontsize=16)
plt.ylabel("Temperatura $[^{\circ}C]$", fontsize=14)
plt.xlabel("Radio $[mm]$ ", fontsize=14)

plt.figure(3)
plt.plot(R_pvc_1 * 1000, Prof_pvc_1 - 273.15, 'g')
plt.title("Perfil de Temperatura vs Radio del $PVC_{1}$", fontsize=16)
plt.ylabel("Temperatura $[^{\circ}C]$", fontsize=14)
plt.xlabel("Radio $[mm]$ ", fontsize=14)

plt.figure(4)
plt.plot(R_pvc_2 * 1000, Prof_pvc_2 - 273.15, 'm')
plt.title("Perfil de Temperatura vs Radio del $PVC_{2}$", fontsize=16)
plt.ylabel("Temperatura $[^{\circ}C]$", fontsize=14)
plt.xlabel("Radio $[mm]$ ", fontsize=14)

plt.figure(5)
plt.plot(R_copp * 1000, (Prof_copper2 / 1e3) + mod_T - 273.15)
plt.title("Perfil de Temperatura vs Radio del Cobre sin Recubrimiento", fontsize=16)
plt.ylabel("Temperatura $[^{\circ}C]$", fontsize=14)
plt.xlabel("Radio $[mm]$ ", fontsize=14)

plt.show()
