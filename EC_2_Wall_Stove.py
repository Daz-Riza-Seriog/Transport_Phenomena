# Code made for Sergio Andrés Díaz Ariza
# 05 March 2021
# License MIT
# Transport Phenomena: Python Program-Case of Study 1-Stove Wall

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# Analysis in basis to Thermal Resistance Network

def Rconv1(h1,As):
    # h = [W/m^2*K]
    # As = [m^2]
    y = 1/h1*As
    return y

def Rconv2(h2,As):
    # h = [W/m^2*K]
    # As = [m^2]
    y = 1/h2*As
    return y

def K_prom_Brick(T1,T2,Tprom,K1,K2): # Calculus of K prom
    m = (K2-K1)/(T2-T1)
    y = m*(Tprom-T1)+K1
    return y

def Rcond_Brick(L1,K1,As):
    # h = [W/m^3*K]
    # As = [m^2]
    y = L1/(K1*As)
    return y

def K_prom_Min_Wall(T1,T2,Tprom,K1,K2): # Calculus of K prom
    m = (K2-K1)/(T2-T1)
    y = m*(Tprom-T1)+K1
    return y

def Rcond_Min_Wall(L1,K2,As):
    # h = [W/m^3*K]
    # As = [m^2]
    y = L1/(K2*As)
    return y

def Rcond_Al_1100(L1,K3,As):
    # h = [W/m^3*K]
    # As = [m^2]
    y = L1/(K3*As)
    return y

def Emissivity_Al_prom(T1,T2,E1,E2,Tprom):
    m = (E2 - E1) / (T2 - T1)
    y = m * (Tprom - T1) + E1
    return y

# Here we define the boundary condition 4 and determine T3
def B_C_4(T_sup4,T_surr2,h_conv2,epsilon_Al,As):
    h_comb = h_conv2 + epsilon_Al*5.670e-8*(T_sup4+T_surr2)*((T_sup4**2)+(T_surr2**2))
    Q = h_comb*As*(T_sup4-T_surr2)
    return Q

def B_C_3(Tsup_2,T_sup4,Qsup4,Rcond4,K2,As):
    L2 = (((Tsup_2-T_sup4)/Qsup4)-Rcond4)*(K2*As)
    return L2

def B_C_2(Tsup_2,Tsurr_1,Qsup4,Rconv1,K2,As):
    L1 = (((Tsurr_1 - Tsup_2) / Qsup4) - Rconv1) * (K2 * As)
    return L1

def B_C_1(L1,hconv1,K1,Tsurr_1,Tsup_2):
    C2 = ((L1*hconv1*Tsurr_1)-(K1*Tsup_2))/(-K1+L1*hconv1)
    return C2

def C1(Tsup_2,Tsup_1,L1):
    C1 =(Tsup_2-Tsup_1)/L1
    return C1

def T_z(C1,C2,z):
    T = C1*z + C2
    return T

T_surr2_range = np.arange(15,35,0.1)
Z = np.linspace(0,1,200)
# L_Al_110 = np.arange(0,1,0.001) # Variation of thickness of the Aluminium layer -- Is fixed "caliber 14"

R1 = Rconv1(30,1)
K_Brick = K_prom_Brick(350+273.15,900+273.15,((350+900)/2)+273.15,0.18,0.31)
K_Min_Wall = K_prom_Min_Wall(40+273.15,315+273.15,((40+315)/2)+273.15,0.04,0.125)
R4 = Rcond_Al_1100(2.10/1000,222,1)
R5 = Rconv2(30,1)

Emissivity_Al = Emissivity_Al_prom(800,1400,0.65,0.45,35)
Q_layer_4 = B_C_4(35+273.15,T_surr2_range+273.15,30,Emissivity_Al,1)
L_layer_2 = B_C_3(700,35,Q_layer_4,R4,K_Min_Wall,1)
L_layer_1 = B_C_2(700,1500,Q_layer_4,R1,K_Brick,1)
C2 = B_C_1(L_layer_1,30,K_Brick,1500,700)
C1 = C1(700,C2,L_layer_1)
T = T_z(C1,C2,Z)

# Temperature profile vs Thickness Wall
T_15 = T_z(C1[0],C2[0],Z)
T_20 = T_z(C1[50],C2[50],Z)
T_25 = T_z(C1[100],C2[100],Z)
T_30 = T_z(C1[150],C2[150],Z)
T_35 = T_z(C1[199],C2[199],Z)


plt.figure(1)
plt.plot([350,900],[0.18,0.31],linestyle='--')
plt.title("Estimacion Conductividad Termica Promedio\n Ladrillo Refractario $[K_{sup1}]$ ",fontsize= 16)
plt.ylabel("Conductividad Termica  $K_{cond}$  $[W/m*K]$",fontsize= 14)
plt.xlabel("Temperatura $[K]$ ",fontsize= 14)

plt.figure(2)
plt.plot([40,315],[0.04,0.125],linestyle='--')
plt.title("Estimacion Conductividad Termica Promedio\n Lana Mineral $[K_{sup2}]$ ",fontsize= 16)
plt.ylabel("Conductividad Termica  $K_{cond}$  $[W/m*K]$",fontsize= 14)
plt.xlabel("Temperatura $[K]$ ",fontsize= 14)

# Temperature profile in Variation of Surrounding temperature
plt.figure(3)
plt.plot(np.linspace(0,1,200),T_15,label='$T_{\infty2}$=15 $[^{\circ}C]$')
plt.plot(np.linspace(0,1,200),T_20,label='$T_{\infty2}$=20 $[^{\circ}C]$')
plt.plot(np.linspace(0,1,200),T_25,label='$T_{\infty2}$=25 $[^{\circ}C]$')
plt.plot(np.linspace(0,1,200),T_30,label='$T_{\infty2}$=30 $[^{\circ}C]$')
plt.plot(np.linspace(0,1,200),T_35,label='$T_{\infty2}$=35 $[^{\circ}C]$')
plt.title("Variacion de Temperatura vs Ancho del Muro\npara $T_{\infty2}$ dadas ",fontsize= 16)
plt.ylabel("Temperatura $[^{\circ}C]$",fontsize= 14)
plt.xlabel("Ancho del Muro $[m]$ ",fontsize= 14)
plt.legend(frameon=False)

# Variation of Layer 2 vs Temperature Surrounding 2
plt.figure(4)
plt.plot(T_surr2_range,L_layer_2)
plt.ylim(0,1)
plt.title("Variacion de Temperatura $T_{\infty2}$\nvs\nAncho Lana Mineral",fontsize= 16)
plt.xlabel("Temperatura $[^{\circ}C]$",fontsize= 14)
plt.ylabel("Ancho del Muro $[m]$ ",fontsize= 14)

# Variation of Layer 1 vs Temperature Surrounding 2
plt.figure(5)
plt.plot(T_surr2_range,L_layer_1)
plt.ylim(0.2,5)
plt.title("Variacion de Temperatura $T_{\infty2}$\nvs\nAncho Ladrillo Refractario",fontsize= 16)
plt.xlabel("Temperatura $[^{\circ}C]$",fontsize= 14)
plt.ylabel("Ancho del Muro $[m]$ ",fontsize= 14)


# Variation of Q vs Temperature Surrounding 2
plt.figure(6)
plt.plot(T_surr2_range,Q_layer_4)
plt.title("Variacion de Tasa de Transferecnia de Calor $\.{Q}$\n$vs$\nTemperatura $T_{\infty2}$ ",fontsize= 16)
plt.xlabel("Temperatura $[^{\circ}C]$",fontsize= 14)
plt.ylabel("$\.{Q}$  [W]",fontsize= 14)

plt.show()

# For stablish temperature outside of 25 Celsisus
print("\nFOR 25 CELSISUS\n")
print("$Q_{tot}$:\n",Q_layer_4[100])
print("length layer 2 $[m]$:\n",L_layer_2[100])
print("length layer 1 $[m]$:\n",L_layer_1[100])
print("C2 & T1 $[K]$:\n",C2[100])
print("C1 $[K]$:\n",C1[100])

# For stablish temperature outside of 25 Celsisus
print("\nFOR 15 CELSISUS\n")
print("$Q_{tot}$:\n",Q_layer_4[0])
print("length layer 2 $[m]$:\n",L_layer_2[0])
print("length layer 1 $[m]$:\n",L_layer_1[0])
print("C2 & T1 $[K]$:\n",C2[0])
print("C1 $[K]$:\n",C1[0])

# For a internal T1 temperature
print("\nFOR T1=1500 CELSISUS\n")
position = np.where(np.isclose(C2,1500,1e-4))[0][0]
print("$Q_{tot}$:\n",Q_layer_4[position])
print("length layer 2 $[m]$:\n",L_layer_2[position])
print("length layer 1 $[m]$:\n",L_layer_1[position])
print("C2 & T1 $[K]$:\n",C2[position])
print("C1 $[K]$:\n",C1[position])
