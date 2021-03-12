# Code made for Sergio Andrés Díaz Ariza
# 12 March 2021
# License MIT
# Transport Phenomena: Python Program-Quizz 1.1

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

myImage = Image.open("Qizz1_12.png") #import image with the content of Assigment

def To(K,Dx,Ts,Te,he):
    y = (((-K*Ts)/Dx) + (he*Te))/(he-(K/Dx))
    return y

def hi(K,Dx,Ts,To):
    y = (-K/5*Dx)*(Ts-To)
    return y

Val_To = To(0.75,0.6,845+273.15,35+273.15,50)
Val_hi = hi(0.75,0.6,845+273.15,Val_To)
print("Values of To:\t",Val_To)
print("Values of hi:\t",Val_hi)

myImage.show()