import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras import callbacks
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import pickle
import streamlit as st

import MasCarreras
import Preprocess
import Informatica
import VAE

def Train():

    Data = pd.read_csv('DataSet/DATATEST-2.csv')


    Data = Preprocess.hola(Data)

    # print(Data["UltimaCarrera"].value_counts()[0:10])


    print(Data.shape)

    AllLoss = 0.0
    AllAccuracy = 0.0
    AllVal_Accuracy = 0.0
    AllVal_Loss = 0.0

    NumIterations = 10

    for x in range(0, NumIterations):
    # ret = Informatica.TestModel(Data)
    # ret = MasCarreras.TestModel(Data)
        ret = VAE.TestModel(Data)

        AllLoss += ret['loss']
        AllAccuracy += ret['accuracy']
        AllVal_Loss += ret['val_loss']
        AllVal_Accuracy += ret['val_accuracy']

    print()
    print("____________________RESULTS__________________")
    print()
    print("Training Accuracy mean from ", NumIterations, " iterations: ", AllAccuracy / NumIterations)
    print("Training Loss mean from ", NumIterations, " iterations: ", AllLoss / NumIterations)
    print("Validation Accuracy mean from ", NumIterations, " iterations: ", AllVal_Accuracy / NumIterations)
    print("Validation Loss mean from ", NumIterations, " iterations: ", AllVal_Loss / NumIterations)



def show_predict_page():

    '''
    'Anho', 'Nhermanos', 'HermanoMayor', 'Compras',
    'EscapeRooms', 'Animales', 'Coches', 'Cocinar', 'SalirConAmigos',
    'Fiesta', 'Naturaleza', 'DeportesNaturaleza', 'Viajar', 'HacerVideos',
    'HacerFotos', 'Instrumentos', 'Dibujar', 'Manualidades', 'Escribir',
    'Cantar', 'Bailar', 'Tecnologia', 'Criptomonedas', 'InvertirBolsa',
    'Leer', 'TelMovil', 'Ordenador', 'Television', 'Series', 'Peliculas',
    'VerDeportes', 'EscucharMusica', 'Videojuegos', 'PracticarDeportes',
    'Gimnasio', 'WhatsApp', 'Youtube', 'Twitter', 'Instagram', 'Twitch',
    'TikTok', 'Linkedin', 'Matematicas', 'LenguaLiteratura', 'Ingles',
    'Historia', 'EducacionFisica', 'Fisica', 'Quimica', 'DibujoTecnico',
    'AsignaturaTecnologia', 'Filosofia', 'Biologia', 'LatinGriego',
    'Frances', 'Religion', 'ManejoOrdenador', 'NumIdiomas', 'PrefiereRural',
    'EstudiosRelaccionadosConLosPadres', 'IrseDeEspanha', 'CuidarPersonas',
    'EscucharPersonas', 'CuidarAnimales', 'Sociable', 'Creativo',
    'Organizada', 'EfectoNegativoSangre', 'UltimaCarrera', 'EsMujer',
    'EsNoBinario', 'InteresTecnologiaTalVez', 'InteresTecnologiaNo',
    'PrefiereCiudad', 'PrefiereMaquinas', 'PrefierePersonas'
'''

    st.title("Career recomender")
    st.write("""### Please enter your data """)
    #https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187
    year = st.slider('years old', 0, 100, 18)



    Nhermanos = st.slider('Nhermanos', 0, 100, 18)
    HermanoMayor = st.checkbox('HermanoMayor')
    Compras = st.slider('Compras', 1, 5, 3)
    EscapeRooms = st.slider('EscapeRooms', 1, 5, 3)
    Animales = st.slider('Animales', 1, 5, 3)
    Coches = st.slider('Coches', 1, 5, 3)
    Cocinar = st.slider('Cocinar', 1, 5, 3)
    SalirConAmigos = st.slider('SalirConAmigos', 1, 5, 3)
    Fiesta = st.slider('Fiesta', 1, 5, 3)
    Naturaleza = st.slider('Naturaleza', 1, 5, 3)
    DeportesNaturaleza = st.slider('DeportesNaturaleza', 1, 5, 3)
    Viajar = st.slider('Viajar', 1, 5, 3)
    HacerVideos = st.slider('HacerVideos', 1, 5, 3)
    HacerFotos = st.slider('HacerFotos', 1, 5, 3)
    Instrumentos = st.slider('Instrumentos', 1, 5, 3)
    Dibujar = st.slider('Dibujar', 1, 5, 3)
    Manualidades = st.slider('Manualidades', 1, 5, 3)
    Escribir = st.slider('Escribir', 1, 5, 3)
    Cantar = st.slider('Cantar', 1, 5, 3)
    Bailar = st.slider('Bailar', 1, 5, 3)
    Tecnologia = st.slider('Tecnologia', 1, 5, 3)
    Criptomonedas = st.slider('Criptomonedas', 1, 5, 3)
    InvertirBolsa = st.slider('InvertirBolsa', 1, 5, 3)
    Leer = st.slider('Leer', 1, 5, 3)
    TelMovil = st.slider('TelMovil', 1, 5, 3)
    Ordenador = st.slider('Ordenador', 1, 5, 3)
    Television = st.slider('Television', 1, 5, 3)
    Series = st.slider('Series', 1, 5, 3)
    Peliculas = st.slider('Peliculas', 1, 5, 3)
    VerDeportes = st.slider('VerDeportes', 1, 5, 3)
    EscucharMusica = st.slider('EscucharMusica', 1, 5, 3)
    Videojuegos = st.slider('Videojuegos', 1, 5, 3)
    PracticarDeportes = st.slider('PracticarDeportes', 1, 5, 3)
    Gimnasio = st.slider('Gimnasio', 1, 5, 3)
    WhatsApp = st.slider('WhatsApp', 1, 5, 3)
    Youtube = st.slider('Youtube', 1, 5, 3)
    Twitter = st.slider('Twitter', 1, 5, 3)
    Instagram = st.slider('Instagram', 1, 5, 3)
    Twitch = st.slider('Twitch', 1, 5, 3)
    TikTok = st.slider('TikTok', 1, 5, 3)
    Linkedin = st.slider('Linkedin', 1, 5, 3)
    Matematicas = st.slider('Matematicas', 1, 5, 3)
    LenguaLiteratura = st.slider('LenguaLiteratura', 1, 5, 3)
    Ingles = st.slider('Ingles', 1, 5, 3)
    Historia = st.slider('Historia', 1, 5, 3)
    EducacionFisica = st.slider('EducacionFisica', 1, 5, 3)
    Fisica = st.slider('Fisica', 1, 5, 3)
    Quimica = st.slider('Quimica', 1, 5, 3)
    DibujoTecnico = st.slider('DibujoTecnico', 1, 5, 3)
    AsignaturaTecnologia = st.slider('AsignaturaTecnologia', 1, 5, 3)
    Filosofia = st.slider('Filosofia', 1, 5, 3)
    Biologia = st.slider('Biologia', 1, 5, 3)
    LatinGriego = st.slider('LatinGriego', 1, 5, 3)
    Frances = st.slider('Frances', 1, 5, 3)
    Religion = st.slider('Religion', 1, 5, 3)
    ManejoOrdenador = st.slider('ManejoOrdenador', 1, 5, 3)
    NumIdiomas = st.slider('NumIdiomas', 1, 5, 3)
    PrefiereRural = st.slider('PrefiereRural', 1, 5, 3)
    EstudiosRelaccionadosConLosPadres = st.slider('EstudiosRelaccionadosConLosPadres', 1, 5, 3)
    IrseDeEspanha = st.slider('IrseDeEspanha', 1, 5, 3)
    CuidarPersonas = st.slider('CuidarPersonas', 1, 5, 3)
    EscucharPersonas = st.slider('EscucharPersonas', 1, 5, 3)
    CuidarAnimales = st.slider('CuidarAnimales', 1, 5, 3)
    Sociable = st.slider('Sociable', 1, 5, 3)
    Creativo = st.slider('Creativo', 1, 5, 3)
    Organizada = st.slider('Organizada', 1, 5, 3)
    EfectoNegativoSangre = st.slider('EfectoNegativoSangre', 1, 5, 3)
    EsMujer = st.slider('EsMujer', 1, 5, 3)
    EsNoBinario = st.slider('EsNoBinario', 1, 5, 3)
    InteresTecnologiaTalVez  = st.slider('InteresTecnologiaTalVez', 1, 5, 3)
    InteresTecnologiaNo  = st.slider('InteresTecnologiaNo', 1, 5, 3)
    PrefiereCiudad  = st.slider('PrefiereCiudad', 1, 5, 3)
    PrefiereMaquinas  = st.slider('PrefiereMaquinas', 1, 5, 3)
    PrefierePersonas  = st.slider('PrefierePersonas', 1, 5, 3)

    #mejora: hacer una encuesta o despues de si piensa que es bullying o no y esto se anyade a datos?
    ok = st.button("predict career")
    if ok:
     #preproccessing
     print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
     print(year)



     st.subheader(f"The estimated career is ${year}")


show_predict_page()
#Train()

