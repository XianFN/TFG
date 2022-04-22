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
import PreprocessConNombresBien
import TodasFlexible
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
        ('Por defecto', 'datosXian', 'datosIrene', 'datosZaira', 'datosJavi', 'datosMaite','datosJabe', 'datosMaria','datosPabloAntes', 'datosPabloAhora'))


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
    st.error("IMPORTANTE: Introduce 0 si no la has cursado")


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
    st.subheader("""Parte 5 y última - Algunas preguntas más específicas.""")

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


    option = st.selectbox(
        'Que te gustaría predecir? (Puedes probar las veces que quieras)',
        ('Por defecto', 'Ramas de conocimiento','Personalizable'))
        #('Por defecto', 'Ramas de conocimiento', 'Informatica o otra'))
    st.text("")
    if option == 'Personalizable':

        todasLasCarreras = ['Ingeniería Informática','Biología','Veterinaria','Ingeniería Electrónica','Magisterio de Educación Primaria','Derecho','Enfermería','Lenguas Modernas - Lenguas Clásicas - Filologías','ADE - Administración y Dirección de Empresas','Biotecnología','Ingeniería Aeroespacial','Ciencias de la Actividad Física y del Deporte','Criminología','Información y Documentación','Ciencia y Tecnología de los Alimentos','Magisterio de Educación Infantil','Marketing','Ingeniería de Sistemas de Información','Historia','Ingeniería Mecánica','Fisioterapia','Ingeniería Industrial','Relaciones Laborales y Recursos Humanos','Ingeniería de Telecomunicación (Teleco) y de Sistemas de Comunicación','Turismo','Ciencias Ambientales','Ingeniería Forestal / Ingeniería del Medio Natural','Psicología','Química','Comercio','Educación Social','Ingeniería Agroambiental','Relaciones Internacionales','Economía','Ingeniería de la Energía','Ingeniería Eléctrica','Humanidades','Física','Geografía y Ordenación del Territorio','Historia del Arte','Finanzas y Contabilidad']
        st.caption(
            "Cuanto mas abajo menos precision tienen (Si me envias los datos al final de esta página puedes ayudar a que haya más disponibles ;)")

        container = st.container()


        seleccionarTodas = st.checkbox('Añadir todas')
        if seleccionarTodas:
            carrerasSeleccionasEntreno = container.multiselect(
                'Selecciona entre que carreras quieres predecir.', todasLasCarreras,todasLasCarreras)
        else:
            carrerasSeleccionasEntreno = container.multiselect(
            'Selecciona entre que carreras quieres predecir.', todasLasCarreras)



    st.text("")
    st.text("")
    ok = st.button("Ver resultados")





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
            Data = pd.read_csv(nombreRuta, index_col=None)
            Data = pd.DataFrame(data=Data)
            changeDtype(Data)
            print(Data.shape)

            print(Data)

        if option == 'Ramas de conocimiento':

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
                st.text("")
                st.write("Las ramas de conocimiento son: Artes y Humanidades,Ciencias, Ciencias de la Salud, Ciencias Sociales y Jurídicas , Ingeniería y Arquitectura")
                st.text("")
                st.write("Las ramas de estudios que tienen un mas del 5% de afinidad contigo según la IA son :")


                otros = 0.0;

                for idx, predictCarrera in enumerate(classPredicted[0]):
                    if float(predictCarrera) > 5:
                        carrerasSeleccionadas.append(carreras[idx])
                        carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                        #st.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                    else:
                        otros += round(float(predictCarrera), 2)

                if otros != 0:
                    carrerasSeleccionadas.append("Otras")
                    carrerasSeleccionadasPorcentaje.append(round(otros,2))

                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)
                carrerasSeleccionadasPorcentaje,carrerasSeleccionadas = zip(*sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))
                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)


                for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):

                    st.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")



                with st.expander("Clica para el porcentaje de cada rama"):
                    allPercents =[]
                    for idx, predictCarrera in enumerate(classPredicted[0]):
                        allPercents.append(round(float(predictCarrera), 2))

                    allPercents,carreras = zip(*sorted(zip(allPercents, carreras), reverse=True))
                    for idx, percent in enumerate(allPercents):
                        st.write(f" {percent} % ->  {carreras[idx]} ")


                st.subheader("Gráfico: ")

                source = pd.DataFrame(
                    {"Ramas": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("porcentaje", stack=True),
                    radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                    color="Ramas",
                )

                c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                c1 + c2

            st.markdown(
                "\n\nEstos resultados está calculados analizando las caracteristicas principales del alumnado encuestado \n"
                "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                "priorizar los gustos personales y hacer lo que mas te guste.")
        elif option == 'Personalizable':

            if len(carrerasSeleccionasEntreno) >= 2:
                with st.spinner("Por favor, espera..."):
                    DataEntreno = pd.read_csv('DataSet/DATATEST-ConNombresBien.csv')
                    print("DENTROOO")

                    DataEntreno = PreprocessConNombresBien.preprocessingDataset(DataEntreno)

                    classPredicted = TodasFlexible.TestModel(DataEntreno,carrerasSeleccionasEntreno,Data,todasLasCarreras)


                    print(classPredicted)
                    print("Predict", classPredicted[0])

                    classPredicted = classPredicted * 100

                    print(classPredicted)


                    carrerasSeleccionadas = []
                    carrerasSeleccionadasPorcentaje = []

                    with st.container():
                        st.subheader("Las carreras predecidas son :")

                        otros = 0.0;

                        for idx, predictCarrera in enumerate(classPredicted[0]):
                            if float(predictCarrera) > 5:
                                carrerasSeleccionadas.append(carrerasSeleccionasEntreno[idx])
                                carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                                # st.text(f" {round(float(predictCarrera), 2)} % ->  {carrerasSeleccionasEntreno[idx]} ")
                            else:
                                otros += round(float(predictCarrera), 2)

                        if otros != 0:
                            carrerasSeleccionadas.append("Otras")
                            carrerasSeleccionadasPorcentaje.append(round(otros, 2))

                        print(carrerasSeleccionadas)
                        print(carrerasSeleccionadasPorcentaje)
                        carrerasSeleccionadasPorcentaje, carrerasSeleccionadas = zip(
                            *sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))
                        print(carrerasSeleccionadas)
                        print(carrerasSeleccionadasPorcentaje)

                        for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):
                            st.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")

                        with st.expander("Clica para el porcentaje de cada carrera"):
                            allPercents = []
                            for idx, predictCarrera in enumerate(classPredicted[0]):
                                allPercents.append(round(float(predictCarrera), 2))

                            allPercents, carrerasSeleccionasEntreno = zip(*sorted(zip(allPercents, carrerasSeleccionasEntreno), reverse=True))
                            for idx, percent in enumerate(allPercents):
                                st.write(f" {percent} % ->  {carrerasSeleccionasEntreno[idx]} ")

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

            else:
                st.error("Tienes que seleccionar al menos 2 carreras para poder predecir.")



        else:
            model = load_model('TodasModelo.h5')
            classPredicted = model.predict(Data)
            print(classPredicted)
            print("Predict", classPredicted[0])

            classPredicted = classPredicted*100

            print(classPredicted)

            carreras = ["Ingeniería Informática", "Biología", "Veterinaria", "Ingeniería Eléctrica", "Magisterio de Educación Primaria", "Derecho", "Enfermería",  "Lenguas Modernas - Lenguas Clásicas - Filologías",
                         "ADE - Administración y Dirección de Empresas", "Biotecnología", "Ingeniería Aeroespacial", "Ciencias de la Actividad Física y del Deporte"]
            carrerasSeleccionadas = []
            carrerasSeleccionadasPorcentaje = []

            with st.container():
                st.subheader("Las carreras predecidas son :")

                otros = 0.0;

                for idx, predictCarrera in enumerate(classPredicted[0]):
                    if float(predictCarrera) > 5:
                        carrerasSeleccionadas.append(carreras[idx])
                        carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                        # st.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                    else:
                        otros += round(float(predictCarrera), 2)

                if otros != 0:
                    carrerasSeleccionadas.append("Otras")
                    carrerasSeleccionadasPorcentaje.append(round(otros,2))

                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)
                carrerasSeleccionadasPorcentaje, carrerasSeleccionadas = zip(
                    *sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))
                print(carrerasSeleccionadas)
                print(carrerasSeleccionadasPorcentaje)

                for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):
                    st.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")

                with st.expander("Clica para el porcentaje de cada carrera"):
                    allPercents = []
                    for idx, predictCarrera in enumerate(classPredicted[0]):
                        allPercents.append(round(float(predictCarrera), 2))

                    allPercents, carreras = zip(*sorted(zip(allPercents, carreras), reverse=True))
                    for idx, percent in enumerate(allPercents):
                        st.write(f" {percent} % ->  {carreras[idx]} ")

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

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.info("Por favor leer después de disfrutar de los resultados! ")
    st.info("Solo para gente que haya o esté cursando una carrera universitaria")
    st.write("")



    st.write("")
    st.write(
        " Para poder ayudarme a mejorar esta IA y poder ser mas precisa necesito mas cantidad de datos, si quieres ayudar a estudiantes indecisos , por favor rellena los datos a continuacion, marca la casilla de consentimiento de datos y darle a Enviar.")


    st.markdown("**Hay una pequeña sorpresa si me envías estos datos**!")
    st.write("")
    newCarrera = st.selectbox(
        '¿Cuál es la última carrera que has cursado o estás cursando? (En caso de no encontrar la carrera específica, por favor, escoge la más similar).',
        ('Por defecto', 'ADE - Administración y Dirección de Empresas', 'Animación', 'Antropología', 'Arqueología', 'Arquitectura', 'Arquitectura Técnica / Ingeniería de la Edificación', 'Artes Escénicas', 'Asistencia en Dirección', 'Astronomía y Astrofísica', 'Bellas Artes', 'Bioinformática - Bioestadística - Biología Computacional', 'Biología', 'Bioquímica', 'Biotecnología', 'Ciberseguridad', 'Ciencia e Ingeniería de Datos', 'Ciencia y Tecnología de los Alimentos', 'Ciencias Ambientales', 'Ciencias Biomédicas', 'Ciencias de la Actividad Física y del Deporte', 'Ciencias del Mar', 'Ciencias Experimentales', 'Ciencias Políticas y de la Administración Pública', 'Cine / Cinematografía', 'Comercio', 'Comunicación', 'Comunicación Audiovisual', 'Conservación y Restauración de Bienes', 'Criminología', 'Culturales', 'Danza', 'Derecho', 'Desarrollo de Aplicaciones Web', 'Diseño', 'Diseño de Interiores', 'Diseño de Moda', 'Diseño de Productos', 'Diseño Digital y Multimedia', 'Diseño gráfico', 'Diseño y Desarrollo de videojuegos', 'Economía', 'Educación Social', 'Emprendimiento', 'Enfermería', 'Enología', 'Estadística', 'Estudios Literarios', 'Farmacia', 'Filosofía', 'Finanzas y Contabilidad', 'Fisioterapia', 'Fotografía', 'Física', 'Física y del Deporte', 'Gastronomía y Ciencias Culinarias', 'Genética', 'Geografía', 'Geografía y Ordenación del Territorio', 'Geología', 'Gestión Aeronáutica', 'Grado Abierto en Artes y Humanidades', 'Grado Abierto en Ciencias Sociales y Jurídicas', 'Grado Abierto en Ingeniería y Arquitectura', 'Historia', 'Historia del Arte', 'Hotelería', 'Humanidades', 'Información y Documentación', 'Ingeniería Aeroespacial', 'Ingeniería Agroambiental', 'Ingeniería Alimentaria', 'Ingeniería Ambiental', 'Ingeniería Biomédica', 'Ingeniería Civil', 'Ingeniería de la Automoción', 'Ingeniería de la Energía', 'Ingeniería de las Industrias Agrarias y Alimentarias', 'Ingeniería de los Materiales', 'Ingeniería de Minas', 'Ingeniería de Sistemas Audiovisuales / Sonido e Imagen', 'Ingeniería de Sistemas Biológicos', 'Ingeniería de Sistemas de Información', 'Ingeniería de Tecnología y Diseño Textil', 'Ingeniería de Telecomunicación (Teleco) y de Sistemas de Comunicación', 'Ingeniería Electrónica', 'Ingeniería Eléctrica', 'Ingeniería en Diseño Industrial y Desarrollo de Producto', 'Ingeniería en Organización Industrial', 'Ingeniería en Tecnologías Industriales', 'Ingeniería Forestal / Ingeniería del Medio Natural', 'Ingeniería Física', 'Ingeniería Geológica', 'Ingeniería Geomática y Topografía', 'Ingeniería Industrial', 'Ingeniería Informática', 'Ingeniería Matemática', 'Ingeniería Mecatrónica', 'Ingeniería Mecánica', 'Ingeniería Naval y Oceánica', 'Ingeniería Náutica y Transporte Marítimo', 'Ingeniería Química', 'Ingeniería Robótica', 'Ingeniería Telemática', 'Ingeniería y desarrollo del Software', 'Lenguas Modernas - Lenguas Clásicas - Filologías', 'Lingüística', 'Logopedia', 'Logística y Ciencias del Transporte', 'Magisterio de Educación Infantil', 'Magisterio de Educación Primaria', 'Marketing', 'Matemáticas', 'Medicina', 'Música', 'Nanociencia y Nanotecnología', 'Nutrición Humana y Dietética', 'Odontología', 'Óptica y Optometría', 'Paisajismo', 'Pedagogía', 'Periodismo', 'Piloto y Dirección de Operaciones Aéreas', 'Podología', 'Protocolo y Organización de Eventos', 'Psicología', 'Publicidad y Relaciones Públicas', 'Química', 'Relaciones Internacionales', 'Relaciones Laborales y Recursos Humanos', 'Seguridad y Control de Riesgos', 'Sociología', 'Teología', 'Terapia Ocupacional', 'Trabajo Social', 'Traducción e Interpretación', 'Turismo', 'Veterinaria'))

    satisfaccion = st.slider('¿Qué nivel de satisfacción tienes respecto a haber cursado esta carrera?', 1, 10, 5)
    acabado = st.radio("¿Has acabado esta carrera ?",
                       ["Si",
                        "Aun no",
                        "La he dejado"])
    st.write("")
    acepta = st.checkbox('Acepto el uso de mis datos')
    st.error(
        "IMPORTANTE: Si ya me has enviado los datos a traves de esta pagina, por favor no lo envies mas veces. Perjudicaría a la precision de la IA")

    enviarDatos = st.button("Enviar")

    if enviarDatos:
        if newCarrera == 'Por defecto':
            st.error('Por favor, selecciona la carrera que has cursado')

        elif acepta:
            with st.spinner("Por favor, espera..."):
                dfEnviar = df;
                print(dfEnviar)
                column_Nhermanos = dfEnviar.pop("Nhermanos")
                column_HermanoMayor = dfEnviar.pop("HermanoMayor")
                dfEnviar = pd.concat([pd.Series({'HermanoMayor': column_HermanoMayor}), dfEnviar])
                dfEnviar = pd.concat([pd.Series({'Nhermanos': column_Nhermanos}), dfEnviar])

                for ind, val in dfEnviar.iteritems():
                    print(ind, val)

                DataEnviar = Preprocess.preprocessingInput(dfEnviar)

                print(DataEnviar.shape)
                DataEnviar = DataEnviar.T  # Es necesario transponer el Dataframe
                print(DataEnviar.shape)
                print(DataEnviar)
                print("XIAN")

                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                print("date and time =", dt_string)

                DataEnviar["UltimaCarrera"] = newCarrera
                DataEnviar["Satisfaccion"] = satisfaccion
                DataEnviar["acabado"] = acabado
                DataEnviar["Fecha"] = dt_string
                DataEnviar["nombre"] = "prueba"
                print(DataEnviar.shape)
                print(DataEnviar)



                DataEnviar.to_csv("NewDataSet/NewData.csv", mode='a', header=False)

            st.success("Muchas gracias por tu aportación!")
            st.balloons()





        else:
            st.error('Por favor, acepta las condiciones')



def changeDtype(Data):

    for column in Data:
       Data[column] = Data[column].astype(np.float32)



show_predict_page()
#Train()


