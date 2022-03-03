import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import statistics

import MasCarreras
import Preprocess
import Informatica

print("Hola")

Data = pd.read_csv('DataSet/DATATEST-2.csv')

Data = Preprocess.hola(Data)

print(Data["UltimaCarrera"].value_counts()[0:10])


print(Data.shape)

AllAccuracy = []

for x in range(0, 5):
    AllAccuracy.append(Informatica.TestModel(Data))
    #AllAccuracy.append(MasCarreras.TestModel(Data))


print(AllAccuracy)
print(statistics.mean(AllAccuracy))
#Informatica.TestModel(Data)
#MasCarreras.TestModel(Data)

#DATOS
''''
Informatica, best: 0.8827956914901733



'''
