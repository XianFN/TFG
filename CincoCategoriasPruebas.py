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

    artes_humanidades = y['UltimaCarrera'].map({"ArtHum": 1}).fillna(0)
    ciencias = y['UltimaCarrera'].map({"Ciencias": 1}).fillna(0)
    ciencias_salud = y['UltimaCarrera'].map({"CienciasSalud": 1}).fillna(0)
    ciencias_sociales_juridicas = y['UltimaCarrera'].map({"CienciasSociales": 1}).fillna(0)
    ingenieria_arquitectura = y['UltimaCarrera'].map({"IngArq": 1}).fillna(0)


    CarrOtra = y['UltimaCarrera'].map({"ArtHum": 0, "Ciencias": 0, "CienciasSalud": 0,"CienciasSociales": 0 , "IngArq": 0 }).fillna(1)

    #CarrOtra = [CarrInformatica or CarrBiologia or CarrVeterinaria or CarrElectronica or CarrMagisterio or CarrDerecho
     #           or CarrEnfermeria or CarrFilologia or CarrADE or CarrBiotecn or CarrAeroesp or CarrDeporte]

    y["artes_humanidades"] = artes_humanidades
    y["ciencias"] = ciencias
    y["ciencias_salud"] = ciencias_salud
    y["ciencias_sociales_juridicas"] = ciencias_sociales_juridicas
    y["ingenieria_arquitectura"] = ingenieria_arquitectura
    y["otra"] = CarrOtra




    y.drop('UltimaCarrera', inplace=True, axis=1)

    changeDtype(y)

   # printColumnValues(y)



    #TODO CAMBIAR!!! Cambiar tambien en anteriores, no se están borrando los de otros, se dejan, MAL

    ''''
    print(y.shape)
    newy = y[y["otra"] == 0]
    print(newy)

    print(newy.shape)
    '''
    return y



def TestModel(Data):






    X_train = Data








    y_train = X_train.UltimaCarrera




    y_train = separarCarreras(y_train)




    X_train.pop("UltimaCarrera")

    print(y_train.value_counts()[0:10])

    print("XIAN ANTES")
    print(X_train.shape)
    print(y_train.shape)

    X_train = X_train[y_train["otra"] == 0]
    y_train = y_train[y_train["otra"] == 0]

    y_train.drop('otra', inplace=True, axis=1)

    print(X_train.shape)
    print(y_train.shape)






    print('YCOLUMNS')
    print(y_train.columns)
    input = len(X_train.columns)

    model = keras.Sequential([
        layers.Dense(75, input_shape=[input]),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(5, activation='softmax'),
    ])

    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,
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
        epochs=500,
        callbacks=[early_stopping],
    )

    history_df = pd.DataFrame(history.history)


    model.save('CincoCategoriasModelo.h5')


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
    print()
    print("GUARDADO CORRECTAMENTE")
    print()
    print()
    print("____________________RESULTS__________________")
    print()
    print("Training Accuracy mean from ", history_df['accuracy'])
    print("Training Loss mean from ", history_df['loss'])


    #TODO preguntar, se deberia pillar el ultimo resultado, o los min. max


'''
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return score[1]
'''
