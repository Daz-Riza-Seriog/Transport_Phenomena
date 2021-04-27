# Code made for Sergio Andrés Díaz Ariza
# 12 Abril 2021
# License MIT
# Transport Phenomena: Python Program-Assessment 4.3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize

sns.set()


class r_cri:

    def R_critic(self, K_wire, h_conve):
        r = K_wire / h_conve
        return r


class Q:

    def Q_system(self, T_surfa_wire, T_enviro, Radius_wire, Thck_R_cri, K_insul, h_conv):
        T_su = T_surfa_wire  # [K]
        T_en = T_enviro  # [K]
        R_w = Radius_wire  # [m]
        R_c = Thck_R_cri  # [m]
        K = K_insul  # [W/m*K]
        h = h_conv  # [W/m^2*K]
        Q = (T_su - T_en) / (((np.log((R_w + R_c) / R_w)) / (2 * np.pi * R_c * K)) + (1 / (h * 2 * np.pi * (R_w + R_c))))
        return Q


R_cri = r_cri()
Q_sys = Q()

R_ciritic = R_cri.R_critic(1.4, 15)
Q_system = Q_sys.Q_system(150+273.15, 25+273.15, 0.02, R_ciritic, 1.4, 15)
print(R_ciritic)
print(Q_system)

########################################################################################################
#   POINT 2

