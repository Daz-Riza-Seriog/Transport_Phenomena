# Code made for Sergio Andrés Díaz Ariza
# 06 March 2021
# License MIT
# Transport Phenomena: Python Program-Assignment 2.3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

def As(D): # Here we determine the superficial Area of Cylinder
    # D = [m] Diameter
    return np.pi*D

def h_conv(Q_conv,Area,T_s,T_surr):
    # Q_conv = [W]
    # T_s, T_surr = [K] or [Celsius]
    hconv = Q_conv/(Area*(T_s-T_surr))
    return hconv

# Array of Data --> Fisrst Column Power, Second column ir velocity
Data = np.array([[450,1],[658,2],[983,4],[1507,8],[1963,12]])

Area = As(0.025)

h1 = h_conv(Data[0,0],Area,300+273.15,40+273.15)
h2 = h_conv(Data[1,0],Area,300+273.15,40+273.15)
h4 = h_conv(Data[2,0],Area,300+273.15,40+273.15)
h8 = h_conv(Data[3,0],Area,300+273.15,40+273.15)
h12 = h_conv(Data[4,0],Area,300+273.15,40+273.15)

Data = np.append(Data,np.array([[h1],[h2],[h4],[h8],[h12]]),axis=1) # Put all data in Array
Df_Data = pd.DataFrame(Data, columns = ['Power [W/m]','Velocity [m/s]','Coefficient [W/m^2*K]'])
print(Df_Data)

Range_n = np.arange(0,30,0.000001)
Range_C = np.arange(0,30,0.000001)

def Vel_dependency(C,n,V,h):
    Params = C*(V**n)
    i_array = np.where(np.isclose(Params, h, 1e-5))[0][0]
    Val_n_C = C[i_array]
    return Val_n_C

y1= Vel_dependency(Range_C,Range_n,Df_Data.loc[0,'Velocity [m/s]'],Df_Data.iloc[0,2])
y2= Vel_dependency(Range_C,Range_n,Df_Data.loc[1,'Velocity [m/s]'],Df_Data.iloc[1,2])
y3= Vel_dependency(Range_C,Range_n,Df_Data.loc[2,'Velocity [m/s]'],Df_Data.iloc[2,2])
y4= Vel_dependency(Range_C,Range_n,Df_Data.loc[3,'Velocity [m/s]'],Df_Data.iloc[3,2])
y5= Vel_dependency(Range_C,Range_n,Df_Data.loc[4,'Velocity [m/s]'],Df_Data.iloc[4,2])

Df_Data["Values of C and n"]= pd.Series([y1,y2,y3,y4,y5])

print("\n\tExperimental data with Heat Coefficient Convection & Values C ,n\n",Df_Data)
