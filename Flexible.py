import os
from datetime import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN



def printColumnValues(Data):

    for column in Data:
        print(Data[column].value_counts())

def changeDtype(Data):

    for column in Data:
        Data[column] = Data[column].astype(np.float32)

def separarCarreras(y,carrerasSeleccionadas):

    y = pd.DataFrame(y, columns=["UltimaCarrera"])

    for idx, carrera in enumerate(carrerasSeleccionadas):
        y[carrera] = y.UltimaCarrera == carrera



    y.drop('UltimaCarrera', inplace=True, axis=1)

    changeDtype(y)

    #printColumnValues(y)

    return y
def toHex(num):
    valor = str(num)
    equivalencias = {
        "10": "A",
        "11": "B",
        "12": "C",
        "13": "D",
        "14": "E",
        "15": "F",
    }
    if valor in equivalencias:
        return equivalencias[valor]
    else:
        return valor
def getRuta(carrerasSeleccionadas,todasCarreras):
    ruta = "src/ModelosFlexibles/Modelo_"
    Decimal = 0
    for idx, carrera in enumerate(todasCarreras):
        #print("INDEX", Decimal, (idx + 1), carrera, (4 - ((idx + 1) % 4)))
        if (idx + 1) % 4 == 0:
            if carrera in carrerasSeleccionadas:
                #print("INSIDE 1", Decimal)
                Decimal = Decimal + pow(2, 3)
                #print("DESPUES 2", Decimal)
            ruta += toHex(Decimal)
            Decimal = 0
        else:

            if carrera in carrerasSeleccionadas:
                #print("INSIDE 2", Decimal)
                Decimal = Decimal + pow(2, ((idx) % 4))
               # print("DESPUES 2", Decimal)

    if Decimal != 0:
        ruta += toHex(Decimal)

    now = datetime.now()
    dt_string = now.strftime("_%d-%m-%Y")

    ruta += dt_string
    ruta += ".h5"

    #print(ruta)
    return ruta
def deleteAllOther(Data, carrerasSeleccionadas):


    Data["borrar"] = Data['UltimaCarrera']
    for carrera in carrerasSeleccionadas:
        Data.loc[Data.borrar == carrera, 'borrar'] = 1


    Data.loc[Data.borrar != 1, 'borrar'] = 0

    #print(Data.shape)
    Data = Data[Data["borrar"] == 1]


    #print(Data.shape)
    #print(Data.UltimaCarrera.value_counts()[0:10])
    Data.pop("borrar")

    #print(Data.shape)

    return Data


def TestModel(Data, carrerasSeleccionadas,DataPredict,todasCarreras):

    #print(Data.UltimaCarrera.value_counts()[40:50])
    ruta= getRuta(carrerasSeleccionadas,todasCarreras)
    if os.path.isfile(ruta):
        print("Ya estaba entrenado este modelo : ",ruta, " , se empieza a predecir.")
        model = load_model(ruta)
        return model.predict(DataPredict)

    else:

        Data = deleteAllOther(Data,carrerasSeleccionadas )

        X_train = Data

        y_train = X_train.UltimaCarrera

        y_train = separarCarreras(y_train,carrerasSeleccionadas )



        X_train.pop("UltimaCarrera")
        #print(X_train.shape)
        #print(y_train.shape)



        input = len(X_train.columns)
        output = len(y_train.columns)


        model = keras.Sequential([
            layers.Dense(75, input_shape=[input]),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(output, activation='softmax'),
        ])

        early_stopping = callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.005,
            patience=20,

        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001),
            loss='categorical_crossentropy',
            # sparse_categorical_crossentropy
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=300,
            callbacks=[early_stopping],
            verbose=0
        )

        history_df = pd.DataFrame(history.history)


        model.save(ruta)
        # ['Ingeniería Informática', 'Biología','Veterinaria','Ingeniería Electrónica',
        # 'Magisterio de Educación Primaria','Derecho','Enfermería','Lenguas Modernas - Lenguas Clásicas - Filologías',
        # 'ADE - Administración y Dirección de Empresas', 'Biotecnología']




        print("Se ha entrenado de forma correcta, con una precision de: ", history_df.iloc[-1]['accuracy'])

        return model.predict(DataPredict)