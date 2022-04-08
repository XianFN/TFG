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
from keras.models import load_model

def TestModel(Data):

    X_train = Data.sample(frac=0.75, random_state=200)

    X_test = Data.drop(X_train.index)

    y_train = X_train.UltimaCarrera == "IngenierÃ­a InformÃ¡tica"
    y_test = X_test.UltimaCarrera == "IngenierÃ­a InformÃ¡tica"

    X_test.pop("UltimaCarrera")
    X_train.pop("UltimaCarrera")

    y_test = pd.DataFrame(y_test, columns=["UltimaCarrera"])

    y_test = y_test['UltimaCarrera'].astype(np.float32)

    y_train = pd.DataFrame(y_train, columns=["UltimaCarrera"])

    y_train = y_train['UltimaCarrera'].astype(np.float32)

    # ''' TODO ADASYN
    print(X_train.shape)
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X_train, y_train)
    print(X_res.shape)
    print(y_res.shape)
    # '''

    '''
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
        layers.Dense(1, activation='sigmoid'),
    ])

    early_stopping = callbacks.EarlyStopping(
        # monitor='accuracy',
        monitor='val_binary_accuracy',
        min_delta=0.005,
        patience=20,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        #loss='categorical_crossentropy',
        #sparse_categorical_crossentropy
        #metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=500,
        callbacks=[early_stopping],
    )
    '''
   # model.save('my_model.h5')
    model2 = load_model('my_model.h5')
    score = model2.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    '''


    history_df = pd.DataFrame(history.history)
    history_df = history_df.rename(columns={'binary_accuracy': 'accuracy'})
    history_df = history_df.rename(columns={'val_binary_accuracy': 'val_accuracy'})

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

