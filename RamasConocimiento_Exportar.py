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

    carrerasSeleccionadas = ["ArtHum", "Ciencias", "CienciasSalud","CienciasSociales", "IngArq"]
    Data = deleteAllOther(Data, carrerasSeleccionadas)

    X_train = Data


    y_train = separarCarreras(X_train.UltimaCarrera)

    X_train.pop("UltimaCarrera")


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
