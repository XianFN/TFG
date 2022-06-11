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

    return y

def deleteAllOther(Data, carrerasSeleccionadas):


    Data["borrar"] = Data['UltimaCarrera']
    for carrera in carrerasSeleccionadas:
        Data.loc[Data.borrar == carrera, 'borrar'] = 1


    Data.loc[Data.borrar != 1, 'borrar'] = 0

    Data = Data[Data["borrar"] == 1]

    Data.pop("borrar")

    return Data

def TestModel(Data):

    carrerasSeleccionadas = ["Ingeniería Informática", "Biología", "Veterinaria", "Ingeniería Electrónica",
                             "Magisterio de Educación Primaria", "Derecho", "Enfermería",
                             "Lenguas Modernas - Lenguas Clásicas - Filologías",
                             "ADE - Administración y Dirección de Empresas", "Biotecnología", "Ingeniería Aeroespacial",
                             "Ciencias de la Actividad Física y del Deporte"]

    Data = deleteAllOther(Data, carrerasSeleccionadas)

    X_train = Data.sample(frac=0.75, random_state=200)

    X_test = Data.drop(X_train.index)


    y_train = separarCarreras(X_train.UltimaCarrera, carrerasSeleccionadas)
    y_test = separarCarreras(X_test.UltimaCarrera, carrerasSeleccionadas)

    '''
    yprueba= y_train
    yprueba = y_train.map(
        {"Ciencias de la Actividad FÃ­sica y del Deporte": 1, "IngenierÃ­a InformÃ¡tica": 2, "BiologÃ­a": 3,
         "Veterinaria": 4, "IngenierÃ­a ElectrÃ³nica": 5,
         "Magisterio de EducaciÃ³n Primaria": 6, "Derecho": 7, "EnfermerÃ­a": 8,
         "Lenguas Modernas - Lenguas ClÃ¡sicas - FilologÃ­as": 9, "ADE - AdministraciÃ³n y DirecciÃ³n de Empresas": 10,
         "BiotecnologÃ­a": 11, "IngenierÃ­a Aeroespacial": 12}).fillna(0)
    '''


    X_test.pop("UltimaCarrera")
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
    print('YCOLUMNS')
    print(y_train.columns)
    input = len(X_train.columns)

    model = keras.Sequential([
        layers.Dense(17, input_shape=[input]),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(12, activation='softmax'),
    ])


    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
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
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=300,
        callbacks=[early_stopping],
    )

    history_df = pd.DataFrame(history.history)


    #model.save('TodasModelo.h5')


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
    return history_df.iloc[-1]

