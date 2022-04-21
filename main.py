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
import time
from keras.models import load_model
import altair as alt
from PIL import Image
from datetime import datetime

import CincoCategorias
import CincoCategoriasPruebas
import MasCarreras
import Preprocess
import Informatica
import VAE
import VAEPRUEBAS


def Train():
    #TODO CUIDADOOOO
    #Data = pd.read_csv('DataSet/DATATEST-2.csv')
    Data = pd.read_csv('DataSet/DATATEST_CincoCategorias.csv')

    Data = Preprocess.preprocessingDataset(Data)

    # print(Data["UltimaCarrera"].value_counts()[0:10])

    print(Data.shape)

    AllLoss = 0.0
    AllAccuracy = 0.0
    AllVal_Accuracy = 0.0
    AllVal_Loss = 0.0

    NumIterations = 1

    for x in range(0, NumIterations):
        #ret = Informatica.TestModel(Data)
        # ret = MasCarreras.TestModel(Data)
        #ret = CincoCategorias.TestModel(Data)
        ret = CincoCategoriasPruebas.TestModel(Data)
        #ret = VAE.TestModel(Data)
        #ret = VAEPRUEBAS.TestModel(Data)

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


    st.set_page_config(page_title='EStuDIA',
                       page_icon="🧊",
                       # layout='wide',
                       initial_sidebar_state='collapsed',
                       menu_items={

                       })

    image = Image.open('Imagenes/Logo.png')

    st.image(image)

    st.text("")
    st.subheader("Recomendador de carreras universitarias basado en Inteligencia Artificial\n\n\n")
    st.text("Deberás responder a esta serie de preguntas( 3 min. aproximado).\nDespues, la IA que he entrenado procesa los datos."
            "\nEsta intentará predecir que carreras universitaras se adaptan a tus gustos. \nDado que se trata de una predicción, los resulta"
            "dos puede que no sean exactos")



    st.text("")
    st.text("")
    st.caption("Trabajo de Fin de Grado de Ingeniería Informática. Desarrollado por Xián Filgueiras Nogueira")


    Input = []
    df = pd.Series(data=Input, dtype=np.float32)


    datosPrecargados = st.sidebar.selectbox(
        'Quieres probar con datos precargados?',
        ('Por defecto', 'datosXian', 'datosIrene', 'datosZaira', 'datosJavi', 'datosMaite','datosJabe', 'datosMaria'))

    st.text("")
    st.subheader("""Parte 1 - Indica del 1(nada) al 5(mucho) lo que te gustan los distintos hobbies: """)

    df["Compras"] = st.slider('ir de compras', 1, 5, 1)
    df["EscapeRooms"] = st.slider('Escape Rooms', 1, 5, 1)
    df["Animales"] = st.slider('Estar con animales', 1, 5, 1)
    df["Coches"] = st.slider('Coches', 1, 5, 1)
    df["Cocinar"] = st.slider('Cocinar', 1, 5, 1)
    df["SalirConAmigos"] = st.slider('Salir con tus amigos', 1, 5, 1)
    df["Fiesta"] = st.slider('Salir de fiesta', 1, 5, 1)
    df["Naturaleza"] = st.slider('Pasar tiempo en la naturaleza', 1, 5, 1)
    df["DeportesNaturaleza"] = st.slider('Hacer deporte en la naturaleza', 1, 5, 1)
    df["Viajar"] = st.slider('Viajar', 1, 5, 1)
    df["HacerVideos"] = st.slider('Grabar vídeos', 1, 5, 1)
    df["HacerFotos"] = st.slider('Fotografía', 1, 5, 1)
    df["Instrumentos"] = st.slider('Tocar instrumentos', 1, 5, 1)
    df["Dibujar"] = st.slider('Dibujar', 1, 5, 1)
    df["Manualidades"] = st.slider('Manualidades', 1, 5, 1)
    df["Escribir"] = st.slider('Escribir', 1, 5, 1)
    df["Cantar"] = st.slider('Cantar', 1, 5, 1)
    df["Bailar"] = st.slider('Bailar', 1, 5, 1)
    df["Tecnologia"] = st.slider('Tecnologia', 1, 5, 1)
    df["Criptomonedas"] = st.slider('Criptomonedas', 1, 5, 1)
    df["InvertirBolsa"] = st.slider('Invertir en bolsa', 1, 5, 1)


    st.text("")
    st.subheader("""Parte 2 - ¿Cuántas horas al día de media dedicas a las siguientes actividades?""")
    df["Leer"] = st.slider('Leer libros', 0, 12, 0)
    df["TelMovil"] = st.slider('Teléfono Móvil', 0, 12, 0)
    df["Ordenador"] = st.slider('Ordenador', 0, 12, 0)
    df["Television"] = st.slider('Televisión', 0, 12, 0)
    df["Series"] = st.slider('Ver series', 0, 12, 0)
    df["Peliculas"] = st.slider('Ver peliculas', 0, 12, 0)
    df["Vereportes"] = st.slider('Ver deportes', 0, 12, 0)
    df["EscucharMusica"] = st.slider('Escuchar Música', 0, 12, 0)
    df["Videojuegos"] = st.slider('Videojuegos', 0, 12, 0)
    df["PracticarDeportes"] = st.slider('Practicar deportes', 0, 12, 0)
    df["Gimnasio"] = st.slider('Gimnasio', 0, 12, 0)

    st.text("")
    st.subheader("""Parte 3 - Indica del 1(nada) al 5(mucho) cuánto usas al día las siguientes redes sociales.""")
    df["WhatsApp"] = st.slider('WhatsApp', 1, 5, 1)
    df["Youtube"] = st.slider('Youtube', 1, 5, 1)
    df["Twitter"] = st.slider('Twitter', 1, 5, 1)
    df["Instagram"] = st.slider('Instagram', 1, 5, 1)
    df["Twitch"] = st.slider('Twitch', 1, 5, 1)
    df["TikTok"] = st.slider('TikTok', 1, 5, 1)
    df["Linkedin"] = st.slider('Linkedin', 1, 5, 1)

    st.text("")
    st.subheader("""Parte 4 - Indica del 1(nada) al 5(mucho) cuánto te gustaban en secundaria las siguientes asignaturas.""")
    st.subheader("IMPORTANTE: Introduce 0 si no la has cursado")

    df["Matematicas"] = st.slider('Matemáticas', 0, 5, 0)
    df["LenguaLiteratura"] = st.slider('Lengua y Literatura', 0, 5, 0)
    df["Ingles"] = st.slider('Inglés', 0, 5, 0)
    df["Historia"] = st.slider('Historia', 0, 5, 0)
    df["EducacionFisica"] = st.slider('Educación Física', 0, 5, 0)
    df["Fisica"] = st.slider('Física', 0, 5, 0)
    df["Quimica"] = st.slider('Química', 0, 5, 0)
    df["DibujoTecnico"] = st.slider('Dibujo Técnico', 0, 5, 0)
    df["AsignaturaTecnologia"] = st.slider('Tecnología', 0, 5, 0)
    df["Filosofia"] = st.slider('Filosofía', 0, 5, 0)
    df["Biologia"] = st.slider('Biología', 0, 5, 0)
    df["LatinGriego"] = st.slider('Latín y Griego', 0, 5, 0)
    df["Frances"] = st.slider('Francés', 0, 5, 0)
    df["Religion"] = st.slider('Religión', 0, 5, 0)

    st.text("")
    st.subheader("""Parte 5 y última - Algunas preguntas más expecíficas.""")

    df["Nhermanos"] = st.slider('Número de hermanos (sin contarte a ti).', 0, 10, 0)
    df["HermanoMayor"] = st.checkbox('Selecciona esta opción si eres el mayor de tus hermanos')
    df["NumIdiomas"] = st.slider('¿En cuántos idiomas te defiendes mediadamente contando el nativo?', 1, 6, 1)
    df["ManejoOrdenador"] = st.slider('¿Cuál es tu nivel de manejo del ordenador?', 1, 5, 1)
    df["InteresEnTecnologia"] = st.radio(
        "¿Tienes interés de como funcionan los aparatos que usamos a diario, como el ordenador, la televisión, la radio…?",
        ["Si", "Un poco", "No"])

    df["PrefiereRural"] = st.radio('¿Te gusta más la vida en el rural o en la ciudad?',
                                   ["Rural", "Ciudad", "Indiferente"])
    df["IrseDeEspanha"] = st.radio('¿Te gustaría irte de España en el futuro?', ["Si", "Tal vez", "No"])
    df["CuidarPersonas"] = st.radio('¿Te gusta cuidar de las personas?', ["Si", "Tal vez", "No"])
    df["EscucharPersonas"] = st.radio('¿Te gusta escuchar a las personas?', ["Si", "Tal vez", "No"])
    df["CuidarAnimales"] = st.radio('¿Te gusta cuidar de los animales?', ["Si", "Tal vez", "No"])
    df["PrefiereMaquinasOPersonas"] = st.radio('¿Prefieres trabajar con máquinas o con personas?',
                                               ["Máquinas", "Indiferente", "Personas"])
    df["Sociable"] = st.radio('¿Te consideras sociable?', ["Si", "Tal vez", "No"])
    df["Creativo"] = st.radio('¿Te consideras creativo?', ["Si", "Tal vez", "No"])
    df["Organizada"] = st.radio("¿Eres una persona organizada?",
                                ["Mucho, me gusta también planificar asuntos ajenos o grupales.",
                                 "Bastante, organizo mis asuntos personales.", "Lo mínimo para alcanzar mis objetivos.",
                                 "No es lo mío."])
    df["EfectoNegativoSangre"] = st.radio('¿Tiene algún efecto negativo para ti manejar o ver sangre?',
                                                  ["Si", "Tal vez", "No"])

    st.text("")
    ok = st.button("Ver resultados")

    st.text("")
    st.text("")
    option = st.selectbox(
        'Quieres probar con otras predicciones?',
        ('Por defecto', 'Cinco categorias', 'Informatica o otra'))



    if ok:



        # preproccessing

        print(df)
        column_Nhermanos = df.pop("Nhermanos")
        column_HermanoMayor = df.pop("HermanoMayor")
        df = pd.concat([pd.Series({'HermanoMayor' : column_HermanoMayor}), df])
        df = pd.concat([pd.Series({'Nhermanos': column_Nhermanos}), df])



        for ind, val in df.iteritems():
            print(ind, val)

        Data = Preprocess.preprocessingInput(df)


        print(Data.shape)
        Data = Data.T  # Es necesario transponer el Dataframe
        print(Data.shape)
        print(Data)
        print("XIAN")




        if datosPrecargados != "Por defecto":
            nombreRuta = "Pruebas/"+ datosPrecargados +".csv"
            print("NOMRBEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE       ",nombreRuta)
            Data = pd.read_csv(nombreRuta, index_col=None)
            Data = pd.DataFrame(data=Data)
            changeDtype(Data)
            print(Data.shape)

            print(Data)
        else:
            now = datetime.now()
            print(now)
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            print("date and time =", dt_string)

            rutaGuardar = "SaveData/" + dt_string + ".cvs"
            Data.to_csv(rutaGuardar)

        if option == 'Cinco categorias':

            model = load_model('CincoCategoriasModelo.h5')
            classPredicted = model.predict(Data)
            print(classPredicted)
            print("Predict", classPredicted[0])

            classPredicted = classPredicted * 100

            print(classPredicted)

            carreras = ["Artes y Humanidades", "Ciencias", "Ciencias de la Salud", "Ciencias Sociales y Jurídicas",
                        "Ingeniería y Arquitectura"]
            carrerasSeleccionadas = []
            carrerasSeleccionadasPorcentaje = []

            with st.container():
                st.subheader("Las carreras predecidas son :")

                for idx, predictCarrera in enumerate(classPredicted[0]):
                    if float(predictCarrera) > 5:
                        carrerasSeleccionadas.append(carreras[idx])
                        carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                        st.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)

                st.subheader("Gráfico: ")

                source = pd.DataFrame(
                    {"Carreras": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("porcentaje", stack=True),
                    radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                    color="Carreras",
                )

                c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                c1 + c2

            st.markdown(
                "\n\nEstos resultados está calculados analizando las caracteristicas principales del alumnado encuestado \n"
                "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                "priorizar los gustos personales y hacer lo que mas te guste.")

        elif option == 'Informatica o otra':
            model = load_model('Informatica.h5')
            classPredicted = model.predict(Data)
            print("Predict", classPredicted[0])
            with st.container():
                nombreCarrera = "Informatica" if classPredicted[0] > 0.5 else "Otra"

                porcentaje = round(float(classPredicted[0]) * 100, 2) if nombreCarrera == "Informatica"else 100 - round(float(classPredicted[0]) * 100, 2)
                st.subheader(f"La carrera predecida es : {nombreCarrera} con un {porcentaje}%")


                st.subheader("Gráfico: ")


                carrerasSeleccionadas = ["Informatica", "Otra"]
                carrerasSeleccionadasPorcentaje = [round(float(classPredicted[0]) * 100, 2), 100 - round(float(classPredicted[0]) * 100, 2)]

                source = pd.DataFrame({"Carreras": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("porcentaje", stack=True),
                    radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                    color="Carreras",
                )

                c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                c1 + c2

            st.markdown(
                "\n\nEstos resultados está calculados analizando las caracteristicas principales del alumnado encuestado \n"
                "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                "priorizar los gustos personales y hacer lo que mas te guste.")
        else:
            model = load_model('TodasModelo.h5')
            classPredicted = model.predict(Data)
            print(classPredicted)
            print("Predict", classPredicted[0])

            classPredicted = classPredicted*100

            print(classPredicted)

            carreras = ["Ingeniería Informática", "Biología", "Veterinaria", "Ingeniería Eléctrica", "Magisterio de Educación Primaria", "Derecho", "Enfermería",  "Lenguas Modernas - Lenguas Clásicas - Filologías",
                         "ADE - Administración y Dirección de Empresas", "Biotecnología", "Ingeniería Aeroespacial", "Ciencias de la Actividad Física y del Deporte", "Otra"]
            carrerasSeleccionadas = []
            carrerasSeleccionadasPorcentaje = []

            with st.container():
                st.subheader("Las carreras predecidas son :")

                for idx, predictCarrera in enumerate(classPredicted[0]):
                    if float(predictCarrera) > 10:

                        carrerasSeleccionadas.append(carreras[idx])
                        carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                        st.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)

                st.subheader("Gráfico: ")

                source = pd.DataFrame({"Carreras": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})


                base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("porcentaje", stack=True),
                    radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                    color="Carreras",
                )

                c1 = base.mark_arc(innerRadius=20, stroke="#fff")


                c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")


                c1 + c2


            st.markdown("\n\nEstos resultados está calculados analizando las caracteristicas principales del alumnado encuestado \n"
                    "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                    "priorizar los gustos personales y hacer lo que mas te guste.")

def changeDtype(Data):

    for column in Data:
       Data[column] = Data[column].astype(np.float32)



show_predict_page()
#Train()


