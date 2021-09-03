# Code made for Sergio Andrés Díaz Ariza
# 31 August 2021
# License MIT
# Transport Phenomena: Python Program-Case of Study

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

# Parameters
C_ab_min = 1.2e-5  # [mol/cm^3]
C_a_inf = 4.2e-5  # [mol/cm^3]
alpha = 23  # [cm/s]

# Material B
R_b = 60  # [cm]
D_ab = 1.0e-6  # [cm^2/s]

# Material B
R_c = 50  # [cm]
D_ac = 7.5e-6  # [cm^2/s]

b1 = alpha * C_a_inf
b2 = C_ab_min
b3 = R_c * C_ab_min
b4 = 0


# b5 = 0

def Cascaron(r, C1, C2, Dab):
    Ca = (C1 / (r * Dab)) + C2
    return Ca


def Esf_Porosa(r, C4, Ko, Dac):
    Ca = ((Ko * (r ** 2)) / (6 * Dac)) + C4
    return Ca


B = np.array([b1, b2, b3, b4])
A = np.array([[((alpha / (R_b * D_ab)) - (1 /(R_b ** 2))), alpha, 0, 0], [(1 / (R_c * D_ab)), 1, 0, 0],
              [0, 0, R_c, ((R_c ** 3)/(6 * D_ac))], [(1/(R_c ** 2)), 0, 0, (R_c / 3)]])

# A_inv = np.linalg.inv(A) ---> We can find the solution directly
C = np.linalg.solve(A, B)

print("Variable C1 = {} [mol*cm^3/s]".format(C[0]))
print("Variable C2 = {} [mol]".format(C[1]))
print("Variable C3 = {} []".format(0.0))
print("Variable C4 = {} [mol]".format(C[2]))
print("Variable ko = {} [mol*cm^3/s]".format(C[3]))

# solve for Temperature Profile
r_cas = np.linspace(R_c, R_b, 100)
r_esf = np.linspace(0.000001, R_c, 100)
C_cascaron = Cascaron(r_cas, C[0], C[1], D_ab)
C_esfera = Esf_Porosa(r_esf, C[2], C[3], D_ac)


plt.figure(1)
plt.plot(r_cas, C_cascaron, label='$50<R_{shell}<60$')
plt.plot(r_esf, C_esfera, label='$0.000001<R_{sphere}<50$')
plt.title("Concentration Profile in a Porous Sphere\n with a Shell", fontsize=16)
plt.ylabel("Concentration $[C_A]$", fontsize=14)
plt.xlabel("Radius of the System $[cm]$ ", fontsize=14)
plt.legend()
plt.show()
