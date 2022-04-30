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
from datetime import datetime

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

    #''' con validation
    X_train = Data.sample(frac=0.75, random_state=200)
    X_test = Data.drop(X_train.index)
    y_test = separarCarreras(X_test.UltimaCarrera, carrerasSeleccionadas)
    #'''
    # Sin validation
    #X_train = Data


    y_train = separarCarreras(X_train.UltimaCarrera, carrerasSeleccionadas)


    # con validation
    X_test.pop("UltimaCarrera")

    X_train.pop("UltimaCarrera")
    print(X_train.shape)
    print(y_train.shape)



    input = len(X_train.columns)
    NumIterations = 300
    for x in range(0, NumIterations):

        model = keras.Sequential([
            layers.Dense(17, input_shape=[input]),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(12, activation='softmax'),
        ])

        early_stopping = callbacks.EarlyStopping(
            #monitor='loss',
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
            verbose=0
        )

        history_df = pd.DataFrame(history.history)



    #
        '''
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

        '''



        if history_df['val_accuracy'].iloc[-1] > 0.50:

            ruta = "src/ModelosBuenos/PorDefecto/Accuracy/PorDefecto_"
            precision = float(history_df['val_accuracy'].iloc[-1]) * 100
            ruta+= str(round(float(precision), 3))
            ruta += "%"
            now = datetime.now()
            dt_string = now.strftime("_%m-%Y")
            ruta += dt_string
            ruta += ".h5"
            model.save(ruta)
            print("____________________________________________________________________________________________")
            print("Guardado, con una precision del : ", precision, "% , guardado correctamente en la ruta:  ",ruta)
            print(history_df.iloc[-1])
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

        elif history_df['val_loss'].iloc[-1] < 1.67:

            ruta = "src/ModelosBuenos/PorDefecto/Loss/PorDefecto_"
            loss = float(history_df['val_loss'].iloc[-1])
            ruta += str(round(float(loss), 3))
            now = datetime.now()
            dt_string = now.strftime("_%m-%Y")
            ruta += dt_string
            ruta += ".h5"
            model.save(ruta)
            print("Guardado, con un loss del : ", loss, " , guardado correctamente en la ruta:  ", ruta)
            print(history_df.iloc[-1])
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

       # print(history_df.iloc[-1])



