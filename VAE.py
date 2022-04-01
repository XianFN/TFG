import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

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

    CarrOtra = y['UltimaCarrera'].map({"Ciencias de la Actividad FÃ­sica y del Deporte": 0, "IngenierÃ­a InformÃ¡tica": 0, "BiologÃ­a": 0,"Veterinaria": 0 , "IngenierÃ­a ElectrÃ³nica": 0,
                                    "Magisterio de EducaciÃ³n Primaria": 0 , "Derecho": 0, "EnfermerÃ­a": 0 , "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as": 0, "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas": 0, "BiotecnologÃ­a": 0, "IngenierÃ­a Aeroespacial": 0 , "Ciencias de la Actividad FÃ­sica y del Deporte": 0 }).fillna(1)

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
    y["CarrOtra"] = CarrOtra

    y.drop('UltimaCarrera', inplace=True, axis=1)

    changeDtype(y)

  #  printColumnValues(y)

    return y



def TestModel(Data):

    X_train = Data.sample(frac=0.75, random_state=200)

    X_test = Data.drop(X_train.index)

    y_train = X_train.UltimaCarrera
    y_test = X_test.UltimaCarrera


    y_train =  separarCarreras(y_train)
    y_test = separarCarreras(y_test)


    X_test.pop("UltimaCarrera")
    X_train.pop("UltimaCarrera")



    input = len(X_train.columns)

    model = keras.Sequential([
        layers.Dense(200, input_shape=[input]),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(200),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(13, activation='softmax'),
    ])

    early_stopping = callbacks.EarlyStopping(
        monitor='accuracy',
        min_delta=0.005,
        patience=20,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01),
        loss='categorical_crossentropy',
        # sparse_categorical_crossentropy
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=64,
        epochs=300,
        callbacks=[early_stopping],
    )

    history_df = pd.DataFrame(history.history)

    plt.title("Training and validation loss results")
    sns.lineplot(data=history_df['loss'], label="Training Loss")
    sns.lineplot(data=history_df['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training and validation accuracy results")
    sns.lineplot(data=history_df['accuracy'], label="Training Accuracy")
    sns.lineplot(data=history_df['val_accuracy'], label="Validation Accuracy")
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