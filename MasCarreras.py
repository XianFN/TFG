import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns

def changeDtype(Data):

    for column in Data:
        Data[column] = Data[column].astype(np.float32)

def separarCarreras(y):

    y = pd.DataFrame(y, columns=["UltimaCarrera"])

    CarrInformatica = []
    CarrBiologia = []
    CarrVeterinaria = []
    CarrElectronica = []
    CarrMagisterio = []
    CarrOtra = []

    for ultimaCarrera in y.UltimaCarrera:
        if ultimaCarrera == "IngenierÃ­a InformÃ¡tica":
            CarrInformatica.append(1)
            CarrBiologia.append(0)
            CarrVeterinaria.append(0)
            CarrElectronica.append(0)
            CarrMagisterio.append(0)
            CarrOtra.append(0)
        elif ultimaCarrera == "BiologÃ­a":
            CarrInformatica.append(0)
            CarrBiologia.append(1)
            CarrVeterinaria.append(0)
            CarrElectronica.append(0)
            CarrMagisterio.append(0)
            CarrOtra.append(0)
        elif ultimaCarrera == "Veterinaria":
            CarrInformatica.append(0)
            CarrBiologia.append(0)
            CarrVeterinaria.append(1)
            CarrElectronica.append(0)
            CarrMagisterio.append(0)
            CarrOtra.append(0)
        elif ultimaCarrera == "IngenierÃ­a ElectrÃ³nica":
            CarrInformatica.append(0)
            CarrBiologia.append(0)
            CarrVeterinaria.append(0)
            CarrElectronica.append(1)
            CarrMagisterio.append(0)
            CarrOtra.append(0)
        elif ultimaCarrera == "Magisterio de EducaciÃ³n Primaria":
            CarrInformatica.append(0)
            CarrBiologia.append(0)
            CarrVeterinaria.append(0)
            CarrElectronica.append(0)
            CarrMagisterio.append(1)
            CarrOtra.append(0)
        else:
            CarrInformatica.append(0)
            CarrBiologia.append(0)
            CarrVeterinaria.append(0)
            CarrElectronica.append(0)
            CarrMagisterio.append(0)
            CarrOtra.append(1)

    y["CarrInformatica"] = CarrInformatica
    y["CarrBiologia"] = CarrBiologia
    y["CarrVeterinaria"] = CarrVeterinaria
    y["CarrElectronica"] = CarrElectronica
    y["CarrMagisterio"] = CarrMagisterio
    y["CarrOtra"] = CarrOtra
    y.drop('UltimaCarrera', inplace=True, axis=1)

    changeDtype(y)

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


    #'''
    print(y_train)
    print(y_test)
    print(y_train.dtypes)
    print(y_test.dtypes)
    print(X_test.dtypes)
    print(X_train.dtypes)

    print(y_train.shape)
    print(y_test.shape)
    print(X_test.shape)
    print(X_train.shape)
    #'''

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
        layers.Dense(6, activation='softmax'),
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
    # TODO preguntar, se deberia pillar el ultimo resultado, o los min. max
    return history_df.iloc[-1]