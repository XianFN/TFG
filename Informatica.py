import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

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
        layers.Dense(200, input_shape=[input]),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(200),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        # layers.Dense(1, activation='softmax'),
        layers.Dense(1, activation='sigmoid'),
    ])

    early_stopping = callbacks.EarlyStopping(
        # monitor='accuracy',
        monitor='binary_accuracy',
        min_delta=0.005,
        patience=20,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        #loss='categorical_crossentropy',
        #sparse_categorical_crossentropy
        #metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=512,
        epochs=500,
        callbacks=[early_stopping],
    )



    plt.plot(history.history['binary_accuracy'])
    plt.show()
    plt.plot(history.history['loss'])
    plt.show()

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

    print(("Best Validation Loss: {:0.4f}" + \
           "\nBest Validation Accuracy: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_binary_accuracy'].max()))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
