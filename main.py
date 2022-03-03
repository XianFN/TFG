import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

import MasCarreras
import Preprocess
import Informatica

print("Hola")

Data = pd.read_csv('DataSet/DATATEST-2.csv')

Data = Preprocess.hola(Data)

print(Data["UltimaCarrera"].value_counts()[0:10])


print(Data.shape)



Informatica.TestModel(Data)
#MasCarreras.TestModel(Data)

