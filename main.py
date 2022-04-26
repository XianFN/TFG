import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
import altair as alt
from PIL import Image
from datetime import datetime

from Obsolete import Preprocess
import PreprocessConNombresBien
import Flexible
import PorDefecto


def Train():
    #TODO CUIDADOOOO
    #Data = pd.read_csv('DataSet/DATATEST-2.csv')
    Data = pd.read_csv('DataSet/DATATEST-ConNombresBien.csv')
    #Data = pd.read_csv('DataSet/DATATEST_CincoCategorias.csv')

    #Data = Preprocess.preprocessingDataset(Data)
    Data = PreprocessConNombresBien.preprocessingDataset(Data)

    # print(Data["UltimaCarrera"].value_counts()[0:10])

    #print(Data.shape)

    AllLoss = 0.0
    AllAccuracy = 0.0
    AllVal_Accuracy = 0.0
    AllVal_Loss = 0.0

    NumIterations = 5

    for x in range(0, NumIterations):
        #ret = Informatica.TestModel(Data)
        # ret = MasCarreras.TestModel(Data)
        #ret = RamasConocimiento.TestModel(Data)
        #ret = RamasConocimiento_Exportar.TestModel(Data)
        ret = PorDefecto.TestModel(Data)
        #ret = PorDefecto_Exportar.TestModel(Data)

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

    Input = []
    if 'pageNum' not in st.session_state:
        st.session_state['pageNum'] = 0
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.Series(data=Input, dtype=np.float32)
    if 'Data' not in st.session_state:
        st.session_state['Data'] = pd.Series(data=Input, dtype=np.float32)
    if 'DataToSave' not in st.session_state:
        st.session_state['DataToSave'] = pd.Series(data=Input, dtype=np.float32)

    pageNum = st.session_state.pageNum
    df = st.session_state.df
    DataToSave = st.session_state.DataToSave

    container = st.container()

    st.text("")

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        previous = st.button("Anterior")
    with col3:
        next = st.button("Siguiente")

    if next:
        if pageNum != 3:

            pageNum+= 1

            st.session_state.pageNum= pageNum;
            #print("Sigueinte 2 : ", pageNum)

    elif previous:
        if pageNum != 0:
            pageNum -= 1
            st.session_state.pageNum = pageNum;
            #print("Anterior 2 : ", pageNum)


    if pageNum== 0:

        image = Image.open('Imagenes/Logo.png')

        container.image(image)
        container.text("")
        container.subheader("Recomendador de carreras universitarias basado en Inteligencia Artificial\n\n\n")
        container.write(
            "Esta IA analiza los resultados que introduzcas a continuacion y predice que carreras se adaptan tus gustos y que porcentaje de afinidad."
            " \nDado que se trata de una predicción, los resultados puede que no sean exactos.")
        container.text("")
        container.write("El tiempo estimado para realizar la encuesta son 3 minutos!")

    elif pageNum == 1:

        container.text("")
        container.subheader("""Parte 1 - Indica del 1(nada) al 5(mucho) lo que te gustan los distintos hobbies: """)

        df["Compras"] = container.slider('ir de compras', 1, 5, 1)
        df["EscapeRooms"] = container.slider('Escape Rooms', 1, 5, 1)
        df["Animales"] = container.slider('Estar con animales', 1, 5, 1)
        df["Coches"] = container.slider('Coches', 1, 5, 1)
        df["Cocinar"] = container.slider('Cocinar', 1, 5, 1)
        df["SalirConAmigos"] = container.slider('Salir con tus amigos', 1, 5, 1)
        df["Fiesta"] = container.slider('Salir de fiesta', 1, 5, 1)
        df["Naturaleza"] = container.slider('Pasar tiempo en la naturaleza', 1, 5, 1)
        df["DeportesNaturaleza"] = container.slider('Hacer deporte en la naturaleza', 1, 5, 1)
        df["Viajar"] = container.slider('Viajar', 1, 5, 1)
        df["HacerVideos"] = container.slider('Grabar vídeos', 1, 5, 1)
        df["HacerFotos"] = container.slider('Fotografía', 1, 5, 1)
        df["Instrumentos"] = container.slider('Tocar instrumentos', 1, 5, 1)
        df["Dibujar"] = container.slider('Dibujar', 1, 5, 1)
        df["Manualidades"] = container.slider('Manualidades', 1, 5, 1)
        df["Escribir"] = container.slider('Escribir', 1, 5, 1)
        df["Cantar"] = container.slider('Cantar', 1, 5, 1)
        df["Bailar"] = container.slider('Bailar', 1, 5, 1)
        df["Tecnologia"] = container.slider('Tecnologia', 1, 5, 1)
        df["Criptomonedas"] = container.slider('Criptomonedas', 1, 5, 1)
        df["InvertirBolsa"] = container.slider('Invertir en bolsa', 1, 5, 1)

        container.text("")
        container.subheader("""Parte 2 - ¿Cuántas horas al día de media dedicas a las siguientes actividades?""")
        df["Leer"] = container.slider('Leer libros', 0, 12, 0)
        df["TelMovil"] = container.slider('Teléfono Móvil', 0, 12, 0)
        df["Ordenador"] = container.slider('Ordenador', 0, 12, 0)
        df["Television"] = container.slider('Televisión', 0, 12, 0)
        df["Series"] = container.slider('Ver series', 0, 12, 0)
        df["Peliculas"] = container.slider('Ver peliculas', 0, 12, 0)
        df["Vereportes"] = container.slider('Ver deportes', 0, 12, 0)
        df["EscucharMusica"] = container.slider('Escuchar Música', 0, 12, 0)
        df["Videojuegos"] = container.slider('Videojuegos', 0, 12, 0)
        df["PracticarDeportes"] = container.slider('Practicar deportes', 0, 12, 0)
        df["Gimnasio"] = container.slider('Gimnasio', 0, 12, 0)

        container.text("")
        container.subheader("""Parte 3 - Indica del 1(nada) al 5(mucho) cuánto usas al día las siguientes redes sociales.""")
        df["WhatsApp"] = container.slider('WhatsApp', 1, 5, 1)
        df["Youtube"] = container.slider('Youtube', 1, 5, 1)
        df["Twitter"] = container.slider('Twitter', 1, 5, 1)
        df["Instagram"] = container.slider('Instagram', 1, 5, 1)
        df["Twitch"] = container.slider('Twitch', 1, 5, 1)
        df["TikTok"] = container.slider('TikTok', 1, 5, 1)
        df["Linkedin"] = container.slider('Linkedin', 1, 5, 1)

        container.text("")
        container.subheader(
            """Parte 4 - Indica del 1(nada) al 5(mucho) cuánto te gustaban en secundaria las siguientes asignaturas.""")
        container.error("IMPORTANTE: Introduce 0 si no la has cursado")

        df["Matematicas"] = container.slider('Matemáticas', 0, 5, 0)
        df["LenguaLiteratura"] = container.slider('Lengua y Literatura', 0, 5, 0)
        df["Ingles"] = container.slider('Inglés', 0, 5, 0)
        df["Historia"] = container.slider('Historia', 0, 5, 0)
        df["EducacionFisica"] = container.slider('Educación Física', 0, 5, 0)
        df["Fisica"] = container.slider('Física', 0, 5, 0)
        df["Quimica"] = container.slider('Química', 0, 5, 0)
        df["DibujoTecnico"] = container.slider('Dibujo Técnico', 0, 5, 0)
        df["AsignaturaTecnologia"] = container.slider('Tecnología', 0, 5, 0)
        df["Filosofia"] = container.slider('Filosofía', 0, 5, 0)
        df["Biologia"] = container.slider('Biología', 0, 5, 0)
        df["LatinGriego"] = container.slider('Latín y Griego', 0, 5, 0)
        df["Frances"] = container.slider('Francés', 0, 5, 0)
        df["Religion"] = container.slider('Religión', 0, 5, 0)

        container.text("")
        container.subheader("""Parte 5 y última - Algunas preguntas más específicas.""")

        df["Nhermanos"] = container.slider('Número de hermanos (sin contarte a ti).', 0, 10, 0)
        df["HermanoMayor"] = container.checkbox('Selecciona esta opción si eres el mayor de tus hermanos')
        df["NumIdiomas"] = container.slider('¿En cuántos idiomas te defiendes mediadamente contando el nativo?', 1, 6, 1)
        df["ManejoOrdenador"] = container.slider('¿Cuál es tu nivel de manejo del ordenador?', 1, 5, 1)
        df["InteresEnTecnologia"] = container.radio(
            "¿Tienes interés de como funcionan los aparatos que usamos a diario, como el ordenador, la televisión, la radio…?",
            ["Si", "Un poco", "No"])

        df["PrefiereRural"] = container.radio('¿Te gusta más la vida en el rural o en la ciudad?',
                                       ["Rural", "Ciudad", "Indiferente"])
        df["IrseDeEspanha"] = container.radio('¿Te gustaría irte de España en el futuro?', ["Si", "Tal vez", "No"])
        df["CuidarPersonas"] = container.radio('¿Te gusta cuidar de las personas?', ["Si", "Tal vez", "No"])
        df["EscucharPersonas"] = container.radio('¿Te gusta escuchar a las personas?', ["Si", "Tal vez", "No"])
        df["CuidarAnimales"] = container.radio('¿Te gusta cuidar de los animales?', ["Si", "Tal vez", "No"])
        df["PrefiereMaquinasOPersonas"] = container.radio('¿Prefieres trabajar con máquinas o con personas?',
                                                   ["Máquinas", "Indiferente", "Personas"])
        df["Sociable"] = container.radio('¿Te consideras sociable?', ["Si", "Tal vez", "No"])
        df["Creativo"] = container.radio('¿Te consideras creativo?', ["Si", "Tal vez", "No"])
        df["Organizada"] = container.radio("¿Eres una persona organizada?",
                                    ["Mucho, me gusta también planificar asuntos ajenos o grupales.",
                                     "Bastante, organizo mis asuntos personales.",
                                     "Lo mínimo para alcanzar mis objetivos.",
                                     "No es lo mío."])
        df["EfectoNegativoSangre"] = container.radio('¿Tiene algún efecto negativo para ti manejar o ver sangre?',
                                              ["Si", "Tal vez", "No"])

        container.text("")

        column_Nhermanos = df.pop("Nhermanos")
        column_HermanoMayor = df.pop("HermanoMayor")
        df = pd.concat([pd.Series({'HermanoMayor': column_HermanoMayor}), df])
        df = pd.concat([pd.Series({'Nhermanos': column_Nhermanos}), df])
        Data = Preprocess.preprocessingInput(df)
        #print(Data.shape)
        Data = Data.T  # Es necesario transponer el Dataframe
        #print(Data.shape)
        #print(Data)
        #print("XIAN")

        st.session_state.Data = Data
        newData = Data
        st.session_state.DataToSave = newData

    elif pageNum == 2:

        container.header("Tus resultados ya están listos!")

        container.write("Si estás cursando o has cursado una carrera universitaria podrías ayudar a mejorar la precision y poder predecir con más carreras.")
        container.write("")

        container.write("Si estás dispuesto, rellena los datos a continuacion, marca la casilla de consentimiento de datos y envialos! En caso contrario dale a siguiente")
        container.write("")
        container.markdown("**Hay una pequeña sorpresa si me envías estos datos**!")
        container.markdown("""---""")
        container.write("")
        newCarrera = container.selectbox(
            '¿Cuál es la última carrera que has cursado o estás cursando? (En caso de no encontrar la carrera específica, por favor, escoge la más similar).',
            ('Por defecto', 'ADE - Administración y Dirección de Empresas', 'Animación', 'Antropología', 'Arqueología',
             'Arquitectura', 'Arquitectura Técnica / Ingeniería de la Edificación', 'Artes Escénicas',
             'Asistencia en Dirección', 'Astronomía y Astrofísica', 'Bellas Artes',
             'Bioinformática - Bioestadística - Biología Computacional', 'Biología', 'Bioquímica', 'Biotecnología',
             'Ciberseguridad', 'Ciencia e Ingeniería de Datos', 'Ciencia y Tecnología de los Alimentos',
             'Ciencias Ambientales', 'Ciencias Biomédicas', 'Ciencias de la Actividad Física y del Deporte',
             'Ciencias del Mar', 'Ciencias Experimentales', 'Ciencias Políticas y de la Administración Pública',
             'Cine / Cinematografía', 'Comercio', 'Comunicación', 'Comunicación Audiovisual',
             'Conservación y Restauración de Bienes', 'Criminología', 'Culturales', 'Danza', 'Derecho',
             'Desarrollo de Aplicaciones Web', 'Diseño', 'Diseño de Interiores', 'Diseño de Moda',
             'Diseño de Productos', 'Diseño Digital y Multimedia', 'Diseño gráfico',
             'Diseño y Desarrollo de videojuegos', 'Economía', 'Educación Social', 'Emprendimiento', 'Enfermería',
             'Enología', 'Estadística', 'Estudios Literarios', 'Farmacia', 'Filosofía', 'Finanzas y Contabilidad',
             'Fisioterapia', 'Fotografía', 'Física', 'Física y del Deporte', 'Gastronomía y Ciencias Culinarias',
             'Genética', 'Geografía', 'Geografía y Ordenación del Territorio', 'Geología', 'Gestión Aeronáutica',
             'Grado Abierto en Artes y Humanidades', 'Grado Abierto en Ciencias Sociales y Jurídicas',
             'Grado Abierto en Ingeniería y Arquitectura', 'Historia', 'Historia del Arte', 'Hotelería', 'Humanidades',
             'Información y Documentación', 'Ingeniería Aeroespacial', 'Ingeniería Agroambiental',
             'Ingeniería Alimentaria', 'Ingeniería Ambiental', 'Ingeniería Biomédica', 'Ingeniería Civil',
             'Ingeniería de la Automoción', 'Ingeniería de la Energía',
             'Ingeniería de las Industrias Agrarias y Alimentarias', 'Ingeniería de los Materiales',
             'Ingeniería de Minas', 'Ingeniería de Sistemas Audiovisuales / Sonido e Imagen',
             'Ingeniería de Sistemas Biológicos', 'Ingeniería de Sistemas de Información',
             'Ingeniería de Tecnología y Diseño Textil',
             'Ingeniería de Telecomunicación (Teleco) y de Sistemas de Comunicación', 'Ingeniería Electrónica',
             'Ingeniería Eléctrica', 'Ingeniería en Diseño Industrial y Desarrollo de Producto',
             'Ingeniería en Organización Industrial', 'Ingeniería en Tecnologías Industriales',
             'Ingeniería Forestal / Ingeniería del Medio Natural', 'Ingeniería Física', 'Ingeniería Geológica',
             'Ingeniería Geomática y Topografía', 'Ingeniería Industrial', 'Ingeniería Informática',
             'Ingeniería Matemática', 'Ingeniería Mecatrónica', 'Ingeniería Mecánica', 'Ingeniería Naval y Oceánica',
             'Ingeniería Náutica y Transporte Marítimo', 'Ingeniería Química', 'Ingeniería Robótica',
             'Ingeniería Telemática', 'Ingeniería y desarrollo del Software',
             'Lenguas Modernas - Lenguas Clásicas - Filologías', 'Lingüística', 'Logopedia',
             'Logística y Ciencias del Transporte', 'Magisterio de Educación Infantil',
             'Magisterio de Educación Primaria', 'Marketing', 'Matemáticas', 'Medicina', 'Música',
             'Nanociencia y Nanotecnología', 'Nutrición Humana y Dietética', 'Odontología', 'Óptica y Optometría',
             'Paisajismo', 'Pedagogía', 'Periodismo', 'Piloto y Dirección de Operaciones Aéreas', 'Podología',
             'Protocolo y Organización de Eventos', 'Psicología', 'Publicidad y Relaciones Públicas', 'Química',
             'Relaciones Internacionales', 'Relaciones Laborales y Recursos Humanos', 'Seguridad y Control de Riesgos',
             'Sociología', 'Teología', 'Terapia Ocupacional', 'Trabajo Social', 'Traducción e Interpretación',
             'Turismo', 'Veterinaria'))

        satisfaccion = container.slider('¿Qué nivel de satisfacción tienes respecto a haber cursado esta carrera?', 1, 10, 5)
        acabado = container.radio("¿Has acabado esta carrera ?",
                           ["Si",
                            "Aun no",
                            "La he dejado"])
        container.write("")
        acepta = container.checkbox('Acepto el uso de mis datos')
        container.error(
            "IMPORTANTE: Si ya me has enviado los datos a traves de esta pagina, por favor no lo envies mas veces. Perjudicaría a la precision de la IA")

        enviarDatos = container.button("Enviar")

        if enviarDatos:
            if newCarrera == 'Por defecto':
                container.error('Por favor, selecciona la carrera que has cursado')

            elif acepta:
                with st.spinner("Por favor, espera..."):
                    DataToSave = st.session_state.DataToSave;


                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    print("Guardados los datos de forma correcta, hora: ", dt_string)

                    DataToSave["UltimaCarrera"] = newCarrera
                    DataToSave["Satisfaccion"] = satisfaccion
                    DataToSave["Acabado"] = acabado
                    DataToSave["Fecha"] = dt_string
                    #print(DataToSave.shape)
                    #print(DataToSave)

                    DataToSave.to_csv("NewDataSet/NewData.csv", mode='a', header=False)

                container.success("Muchas gracias por tu aportación!")
                container.balloons()





            else:
                container.error('Por favor, acepta las condiciones')


    elif pageNum == 3:

        datosPrecargados = st.sidebar.selectbox(
            'Quieres probar con datos precargados?',
            ('Por defecto', 'datosXian', 'datosIrene', 'datosZaira', 'datosJavi', 'datosMaite', 'datosJabe',
             'datosMaria', 'datosPabloAntes', 'datosPabloAhora'))

        if datosPrecargados != "Por defecto":

            nombreRuta = "Pruebas/" + datosPrecargados + ".csv"
            Data = pd.read_csv(nombreRuta, index_col=None)
            Data = pd.DataFrame(data=Data)
            changeDtype(Data)
            print("Se han cargado de forma correcta los datos de : ", datosPrecargados)
            #print(Data.shape)

            #print(Data)
        else:
            Data = st.session_state.Data




        container.text("")
        option = container.selectbox(
            'Que te gustaría predecir? (Puedes probar las veces que quieras)',
            ('Por defecto', 'Ramas de conocimiento', 'Personalizable'))
        # ('Por defecto', 'Ramas de conocimiento', 'Informatica o otra'))
        container.text("")
        if option == 'Personalizable':

            todasLasCarreras = ['Ingeniería Informática', 'Biología', 'Veterinaria', 'Ingeniería Electrónica',
                                'Magisterio de Educación Primaria', 'Derecho', 'Enfermería',
                                'Lenguas Modernas - Lenguas Clásicas - Filologías',
                                'ADE - Administración y Dirección de Empresas', 'Biotecnología',
                                'Ingeniería Aeroespacial', 'Ciencias de la Actividad Física y del Deporte',
                                'Criminología', 'Información y Documentación', 'Ciencia y Tecnología de los Alimentos',
                                'Magisterio de Educación Infantil', 'Marketing',
                                'Ingeniería de Sistemas de Información', 'Historia', 'Ingeniería Mecánica',
                                'Fisioterapia', 'Ingeniería Industrial', 'Relaciones Laborales y Recursos Humanos',
                                'Ingeniería de Telecomunicación (Teleco) y de Sistemas de Comunicación', 'Turismo',
                                'Ciencias Ambientales', 'Ingeniería Forestal / Ingeniería del Medio Natural',
                                'Psicología', 'Química', 'Comercio', 'Educación Social', 'Ingeniería Agroambiental',
                                'Relaciones Internacionales', 'Economía', 'Ingeniería de la Energía',
                                'Ingeniería Eléctrica', 'Humanidades', 'Física',
                                'Geografía y Ordenación del Territorio', 'Historia del Arte', 'Finanzas y Contabilidad']


            container2 = container.container()

            seleccionarTodas = container.checkbox('Añadir todas')
            if seleccionarTodas:
                container2.text("")
                carrerasSeleccionasEntreno = container2.multiselect(
                    'Selecciona entre que carreras quieres predecir (Cuanto mas abajo, menos precision).', todasLasCarreras, todasLasCarreras)

            else:
                container2.text("")
                carrerasSeleccionasEntreno = container2.multiselect(
                    'Selecciona entre que carreras quieres predecir (Cuanto mas abajo, menos precision).', todasLasCarreras)


        container.text("")
        container.text("")
        col1, col2, col3 = container.columns(3)

        with col2:
            ok = st.button("Ver resultados")

        container.markdown("""---""")

        if ok:
            Data = Data.drop(['UltimaCarrera','Satisfaccion','Acabado','Fecha'], axis=1, errors='ignore')

            if option == 'Ramas de conocimiento':

                model = load_model('CincoCategoriasModelo.h5')
                classPredicted = model.predict(Data)
                #print(classPredicted)
                #print("Predict", classPredicted[0])

                classPredicted = classPredicted * 100

                carreras = ["Artes y Humanidades", "Ciencias", "Ciencias de la Salud", "Ciencias Sociales y Jurídicas",
                            "Ingeniería y Arquitectura"]
                carrerasSeleccionadas = []
                carrerasSeleccionadasPorcentaje = []

                with st.container():
                    container.text("")
                    container.write(
                        "Las ramas de conocimiento son: Artes y Humanidades,Ciencias, Ciencias de la Salud, Ciencias Sociales y Jurídicas , Ingeniería y Arquitectura")
                    container.text("")
                    container.write("Ramas de estudios predichas: ")

                    otros = 0.0;

                    for idx, predictCarrera in enumerate(classPredicted[0]):
                        if float(predictCarrera) > 5:
                            carrerasSeleccionadas.append(carreras[idx])
                            carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                            # container.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                        else:
                            otros += round(float(predictCarrera), 2)

                    if otros != 0:
                        carrerasSeleccionadas.append("Otras")
                        carrerasSeleccionadasPorcentaje.append(round(otros, 2))

                    #print(carrerasSeleccionadas)
                    #print(carrerasSeleccionadasPorcentaje)
                    carrerasSeleccionadasPorcentaje, carrerasSeleccionadas = zip(
                        *sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))

                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    print("_______________________________")
                    print("Se ha predicho a las ", dt_string, "en la categoría de Ramas de Conocimiento: ")
                    print(carrerasSeleccionadas)
                    print(carrerasSeleccionadasPorcentaje)
                    print("_______________________________")

                    for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):
                        container.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")

                    with container.expander("Clica para el porcentaje de cada rama"):
                        allPercents = []
                        for idx, predictCarrera in enumerate(classPredicted[0]):
                            allPercents.append(round(float(predictCarrera), 2))

                        allPercents, carreras = zip(*sorted(zip(allPercents, carreras), reverse=True))
                        for idx, percent in enumerate(allPercents):
                            st.write(f" {percent} % ->  {carreras[idx]} ")

                    container.subheader("Gráfico: ")

                    source = pd.DataFrame(
                        {"Ramas": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                    base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta("porcentaje", stack=True),
                        radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                        color="Ramas",
                    )

                    c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                    c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                    container.altair_chart(c1+c2, use_container_width=True)

                container.markdown(
                    "\n\nEstos resultados están calculados analizando las características principales del alumnado encuestado \n"
                    "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                    "priorizar los gustos personales y hacer lo que más te guste.")
            elif option == 'Personalizable':

                if len(carrerasSeleccionasEntreno) >= 2:
                    with st.spinner("Por favor, espera..."):
                        DataEntreno = pd.read_csv('DataSet/DATATEST-ConNombresBien.csv')


                        DataEntreno = PreprocessConNombresBien.preprocessingDataset(DataEntreno)

                        classPredicted = Flexible.TestModel(DataEntreno, carrerasSeleccionasEntreno, Data,
                                                                 todasLasCarreras)

                        #print(classPredicted)
                        #print("Predict", classPredicted[0])

                        classPredicted = classPredicted * 100

                        carrerasSeleccionadas = []
                        carrerasSeleccionadasPorcentaje = []

                        with st.container():
                            container.subheader("Las carreras predichas son :")

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

                            #print(carrerasSeleccionadas)
                            #print(carrerasSeleccionadasPorcentaje)
                            carrerasSeleccionadasPorcentaje, carrerasSeleccionadas = zip(
                                *sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))

                            now = datetime.now()
                            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                            print("_______________________________")
                            print("Se ha predicho a las ",dt_string , "en la categoría de Personalizable: ")
                            print(carrerasSeleccionadas)
                            print(carrerasSeleccionadasPorcentaje)
                            print("_______________________________")

                            for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):
                                container.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")

                            with container.expander("Clica para el porcentaje de cada carrera"):
                                allPercents = []
                                for idx, predictCarrera in enumerate(classPredicted[0]):
                                    allPercents.append(round(float(predictCarrera), 2))

                                allPercents, carrerasSeleccionasEntreno = zip(
                                    *sorted(zip(allPercents, carrerasSeleccionasEntreno), reverse=True))
                                for idx, percent in enumerate(allPercents):
                                    st.write(f" {percent} % ->  {carrerasSeleccionasEntreno[idx]} ")

                            container.subheader("Gráfico: ")

                            source = pd.DataFrame(
                                {"Carreras": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                            base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta("porcentaje", stack=True),
                                radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                                color="Carreras",
                            )

                            c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                            c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                            container.altair_chart(c1+c2, use_container_width=True)

                        container.markdown(
                            "\n\nEstos resultados están calculados analizando las características principales del alumnado encuestado \n"
                            "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                            "priorizar los gustos personales y hacer lo que más te guste.")

                else:
                    container.error("Tienes que seleccionar al menos 2 carreras para poder predecir.")



            else:
                model = load_model('TodasModelo.h5')
                classPredicted = model.predict(Data)
                #print(classPredicted)
                #print("Predict", classPredicted[0])

                classPredicted = classPredicted * 100


                carreras = ["Ingeniería Informática", "Biología", "Veterinaria", "Ingeniería Eléctrica",
                            "Magisterio de Educación Primaria", "Derecho", "Enfermería",
                            "Lenguas Modernas - Lenguas Clásicas - Filologías",
                            "ADE - Administración y Dirección de Empresas", "Biotecnología", "Ingeniería Aeroespacial",
                            "Ciencias de la Actividad Física y del Deporte"]
                carrerasSeleccionadas = []
                carrerasSeleccionadasPorcentaje = []

                with st.container():
                    container.subheader("Las carreras predichas son :")

                    otros = 0.0;

                    for idx, predictCarrera in enumerate(classPredicted[0]):
                        if float(predictCarrera) > 5:
                            carrerasSeleccionadas.append(carreras[idx])
                            carrerasSeleccionadasPorcentaje.append(round(float(predictCarrera), 2))
                            # container.text(f" {round(float(predictCarrera), 2)} % ->  {carreras[idx]} ")
                        else:
                            otros += round(float(predictCarrera), 2)

                    if otros != 0:
                        carrerasSeleccionadas.append("Otras")
                        carrerasSeleccionadasPorcentaje.append(round(otros, 2))

                    #print(carrerasSeleccionadas)
                    #print(carrerasSeleccionadasPorcentaje)
                    carrerasSeleccionadasPorcentaje, carrerasSeleccionadas = zip(
                        *sorted(zip(carrerasSeleccionadasPorcentaje, carrerasSeleccionadas), reverse=True))

                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                    print("_______________________________")
                    print("Se ha predicho a las ", dt_string, "en la categoría de Por defecto: ")
                    print(carrerasSeleccionadas)
                    print(carrerasSeleccionadasPorcentaje)
                    print("_______________________________")


                    for idx, percent in enumerate(carrerasSeleccionadasPorcentaje):
                        container.text(f" {percent} % ->  {carrerasSeleccionadas[idx]} ")

                    with container.expander("Clica para el porcentaje de cada carrera"):
                        allPercents = []
                        for idx, predictCarrera in enumerate(classPredicted[0]):
                            allPercents.append(round(float(predictCarrera), 2))

                        allPercents, carreras = zip(*sorted(zip(allPercents, carreras), reverse=True))
                        for idx, percent in enumerate(allPercents):
                            st.write(f" {percent} % ->  {carreras[idx]} ")

                    container.subheader("Gráfico: ")

                    source = pd.DataFrame(
                        {"Carreras": carrerasSeleccionadas, "porcentaje": carrerasSeleccionadasPorcentaje})

                    base = alt.Chart(source).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta("porcentaje", stack=True),
                        radius=alt.Radius("porcentaje", scale=alt.Scale(type="sqrt", zero=True, rangeMin=15)),
                        color="Carreras",
                    )

                    c1 = base.mark_arc(innerRadius=20, stroke="#fff")

                    c2 = base.mark_text(radiusOffset=20).encode(text="porcentaje")

                    container.altair_chart(c1+c2, use_container_width=True)

                container.markdown(
                    "\n\nEstos resultados están calculados analizando las características principales del alumnado encuestado \n"
                    "Está pensado para poder ayudar a estudiantes indecisos. Pero siempre se debería "
                    "priorizar los gustos personales y hacer lo que más te guste.")

        container.write("")
        container.write("")







    st.text("")
    st.text("")
    st.caption("Trabajo de Fin de Grado de Ingeniería Informática. Desarrollado por Xián Filgueiras Nogueira")



def changeDtype(Data):

    for column in Data:
       Data[column] = Data[column].astype(np.float32)



#show_predict_page()
Train()


