import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN

def getMinAndMaxForReturn(history):

#NOT USED
    ret = []
    ret['accuracy']= history['accuracy'].max()
    ret['accuracy'] = history['accuracy'].max()
    ret['accuracy'] = history['accuracy'].max()
    ret['accuracy'] = history['accuracy'].max()

    return ret

def printColumnValues(Data):

    for column in Data:
        print(Data[column].value_counts())

def changeDtype(Data):

    for column in Data:
        Data[column] = Data[column].astype(np.float32)

def separarCarreras(y):

    y = pd.DataFrame(y, columns=["UltimaCarrera"])

    CarrInformatica = y['UltimaCarrera'].map({"IngenierÃ­a InformÃ¡tica": 1}).fillna(0)
    CarrBiologia = y['UltimaCarrera'].map({"BiologÃ­a": 1}).fillna(0)
    CarrVeterinaria = y['UltimaCarrera'].map({"Veterinaria": 1}).fillna(0)
    CarrElectronica = y['UltimaCarrera'].map({"IngenierÃ­a ElectrÃ³nica": 1}).fillna(0)
    CarrMagisterio = y['UltimaCarrera'].map({"Magisterio de EducaciÃ³n Primaria": 1}).fillna(0)
    CarrDerecho = y['UltimaCarrera'].map({"Derecho": 1}).fillna(0)
    CarrEnfermeria = y['UltimaCarrera'].map({"EnfermerÃ­a": 1}).fillna(0)
    CarrFilologia = y['UltimaCarrera'].map({"Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as": 1}).fillna(0)
    CarrADE = y['UltimaCarrera'].map({"ADE - AdministraciÃ³n y DirecciÃ³n de Empresas": 1}).fillna(0)
    CarrBiotecn = y['UltimaCarrera'].map({"BiotecnologÃ­a": 1}).fillna(0)
    CarrAeroesp = y['UltimaCarrera'].map({"IngenierÃ­a Aeroespacial": 1}).fillna(0)
    CarrDeporte = y['UltimaCarrera'].map({"Ciencias de la Actividad FÃ­sica y del Deporte": 1}).fillna(0)


    #CarrOtra = [CarrInformatica or CarrBiologia or CarrVeterinaria or CarrElectronica or CarrMagisterio or CarrDerecho
     #           or CarrEnfermeria or CarrFilologia or CarrADE or CarrBiotecn or CarrAeroesp or CarrDeporte]
    # TODO PREGUNTAR MAITE carrOtra

    y["CarrInformatica"] = CarrInformatica
    y["CarrBiologia"] = CarrBiologia
    y["CarrVeterinaria"] = CarrVeterinaria
    y["CarrElectronica"] = CarrElectronica
    y["CarrMagisterio"] = CarrMagisterio
    y["CarrDerecho"] = CarrDerecho
    y["CarrEnfermeria"] = CarrEnfermeria
    y["CarrFilologia"] = CarrFilologia
    y["CarrADE"] = CarrADE
    y["CarrBiotecn"] = CarrBiotecn
    y["CarrAeroesp"] = CarrAeroesp
    y["CarrDeporte"] = CarrDeporte



   #print(CarrOtra)


    y.drop('UltimaCarrera', inplace=True, axis=1)

    changeDtype(y)

  #  printColumnValues(y)

    return y

def deleteAllOther(Data):

    print("Se viene borradura")
    print(Data.shape)

    Data['UltimaCarrera'] = Data['UltimaCarrera'].map(
        {"Ciencias de la Actividad FÃ­sica y del Deporte": "Ciencias de la Actividad FÃ­sica y del Deporte", "IngenierÃ­a InformÃ¡tica": "IngenierÃ­a InformÃ¡tica", "BiologÃ­a": "BiologÃ­a",
         "Veterinaria":  "Veterinaria", "IngenierÃ­a ElectrÃ³nica": "IngenierÃ­a ElectrÃ³nica",
         "Magisterio de EducaciÃ³n Primaria": "Magisterio de EducaciÃ³n Primaria", "Derecho": "Derecho", "EnfermerÃ­a": "EnfermerÃ­a",
         "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as": "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as", "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas": "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas",
         "BiotecnologÃ­a": "BiotecnologÃ­a", "IngenierÃ­a Aeroespacial": "IngenierÃ­a Aeroespacial ",
         "Ciencias de la Actividad FÃ­sica y del Deporte": "Ciencias de la Actividad FÃ­sica y del Deporte"}).fillna("OTRA")

    print(Data.UltimaCarrera)
    Data = Data[~Data.UltimaCarrera.str.contains("OTRA")]
    print(Data.UltimaCarrera)

    print(Data.shape)

    return Data


def TestModel(Data):

    Data = deleteAllOther(Data)

    X_train = Data

    y_train = X_train.UltimaCarrera


    yprueba= y_train
    yprueba = y_train.map(
        {"Ciencias de la Actividad FÃ­sica y del Deporte": 1, "IngenierÃ­a InformÃ¡tica": 2, "BiologÃ­a": 3,
         "Veterinaria": 4, "IngenierÃ­a ElectrÃ³nica": 5,
         "Magisterio de EducaciÃ³n Primaria": 6, "Derecho": 7, "EnfermerÃ­a": 8,
         "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as": 9, "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas": 10,
         "BiotecnologÃ­a": 11, "IngenierÃ­a Aeroespacial": 12}).fillna(0)

    y_train = separarCarreras(y_train)




    X_train.pop("UltimaCarrera")
    print(X_train.shape)
    print(y_train.shape)

    ''' TODO ADASYN

    print(X_train.shape)
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X_train, yprueba)
    print(X_res.shape)
    print(y_res.shape)
    y_res = y_res.map(
        {1: "Ciencias de la Actividad FÃ­sica y del Deporte", 2: "IngenierÃ­a InformÃ¡tica", 3: "BiologÃ­a",
         4: "Veterinaria", 5: "IngenierÃ­a ElectrÃ³nica",
         6 : "Magisterio de EducaciÃ³n Primaria", 7: "Derecho", 8: "EnfermerÃ­a",
         9 : "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as", 10 : "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas",
         11 : "BiotecnologÃ­a", 12 : "IngenierÃ­a Aeroespacial",
         13 : "Ciencias de la Actividad FÃ­sica y del Deporte"}).fillna(0)

    y_train = separarCarreras(y_res)
    X_train = X_res


    '''

    input = len(X_train.columns)


    model = keras.Sequential([
        layers.Dense(75, input_shape=[input]),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(32),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(12, activation='softmax'),
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
    )

    history_df = pd.DataFrame(history.history)


    model.save('TodasModelo.h5')


    plt.title("Training and validation loss results")
    sns.lineplot(data=history_df['loss'], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training and validation accuracy results")
    sns.lineplot(data=history_df['accuracy'], label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


    print(history_df.iloc[-1])
    #TODO preguntar, se deberia pillar el ultimo resultado, o los min. max
    return history_df.iloc[-1]

'''
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score[1]
'''
