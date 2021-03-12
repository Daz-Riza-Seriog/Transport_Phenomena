# Code made for Sergio Andrés Díaz Ariza
# 12 March 2021
# License MIT
# Transport Phenomena: Python Program-Quizz 1.1

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

myImage = Image.open("Qizz1_1.png") #import image with the content of Assigment

def C1(he,Te,Ti,K,R2,hi,R1):
    y = he*(Te-Ti)/(K/R2+he*K/(hi*R1)+he*np.log(R2/R1))
    return y

def C2(Ti,hi,C1,R1,K):
    y = Ti + C1*(K/(hi*R1)-np.log(R1))
    return y

def Qin(hi,Ti,R1,C1,C2,K): #Carefull is by conduction not convection
    T_r = C1*np.log(R1)+C2
    y = (hi*(Ti-T_r))/-K
    return y

valC1 = C1(50,30,240,25,0.3,125,0.2)
valC2 = C2(240,125,valC1,0.2,25)
valQin =Qin(125,240,0.2,valC1,valC2,25)
print("Value of C1:\t", valC1)
print("Value of C2:\t", valC2)
print("Value of Qin:\t", valQin)

myImage.show()