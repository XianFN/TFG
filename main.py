import tensorflow as tf
import numpy as np
import pandas as pd

import Preprocess

print("Hola")

Data = pd.read_csv('DataSet/DATATEST.csv')

Data = Preprocess.hola(Data)

#print(Data.describe)
