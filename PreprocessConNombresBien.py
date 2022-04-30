import tensorflow as tf
import numpy as np
import pandas as pd

def preprocessingDataset(Data):

    #print(Data.columns)
    #print(Data.shape)

    missing_values_count = Data.isnull().sum()

    # look at the # of missing points in the first ten columns
    #print(missing_values_count[0:10])


    Data.pop("MarcaTemporal")
    Data.pop("ConsentimientoDatos")
    Data = Data.rename(columns={'Tecnologia.1': 'AsignaturaTecnologia'})


    for ind, year in enumerate(Data.Anho):
        Data.Anho[ind] = 2022 - Data.Anho[ind]


    EsMujer = {}
    EsNoBinario = {}
    InteresTecnologiaTalVez = {}
    InteresTecnologiaNo = {}
    PrefiereRural_ = {}
    PrefiereCiudad = {}
    PrefierePersonas = {}
    PrefiereMaquinas = {}

    for ind, persona in Data.iterrows():

        EsMujer[ind] = 1 if persona['Genero'] == "Mujer" else 0
        EsNoBinario[ind] = 1 if persona['Genero'] == "Prefiero no contestar" or persona['Genero'] == "Género no binario" else 0
        Data["Nhermanos"][ind] = 6 if persona["Nhermanos"] == "+5" else persona["Nhermanos"]
        Data["HermanoMayor"][ind] = 1 if persona["HermanoMayor"] else 0
        InteresTecnologiaTalVez[ind] = 1 if persona['InteresEnTecnologia'] == "Un poco" else 0
        InteresTecnologiaNo[ind] = 1 if persona['InteresEnTecnologia'] == "No" else 0
        PrefiereRural_[ind] = 1 if persona['PrefiereRural'] == "Rural" else 0
        PrefiereCiudad[ind] = 1 if persona['PrefiereRural'] == "Ciudad" else 0





        asignaturas = ['Matematicas', 'LenguaLiteratura', 'Ingles', 'Historia', 'EducacionFisica', 'Fisica', 'Quimica',
                       'DibujoTecnico', 'AsignaturaTecnologia', 'Filosofia', 'Biologia', 'LatinGriego', 'Frances',
                       'Religion']
        for asignatura in asignaturas:
            Data[asignatura][ind] = 0 if persona[asignatura] == "No la cursé" else persona[asignatura]

        columnas = ['EstudiosRelaccionadosConLosPadres', 'IrseDeEspanha', 'CuidarPersonas', 'EscucharPersonas',
                    'CuidarAnimales', 'Sociable', 'Creativo', 'EfectoNegativoSangre']
        for columna in columnas:
            Data[columna][ind] = 0 if persona[columna] == "No" else 1

        if persona.Organizada == "No es lo mío.":
            Data["Organizada"][ind] = 0
        elif persona.Organizada == "Lo mínimo para alcanzar mis objetivos.":
            Data["Organizada"][ind] = 1
        elif persona.Organizada == "Bastante, organizo mis asuntos personales.":
            Data["Organizada"][ind] = 2
        elif persona.Organizada == "Mucho, me gusta también planificar asuntos ajenos o grupales.":
            Data["Organizada"][ind] = 3

        PrefierePersonas[ind] = 1 if persona['PrefiereMaquinasOPersonas'] == "Personas" else 0
        PrefiereMaquinas[ind] = 1 if persona['PrefiereMaquinasOPersonas'] == "Máquinas" else 0



    Data["EsMujer"] = pd.Series(EsMujer)
    Data["EsNoBinario"] = pd.Series(EsNoBinario)
    Data["InteresTecnologiaTalVez"] = pd.Series(InteresTecnologiaTalVez)
    Data["InteresTecnologiaNo"] = pd.Series(InteresTecnologiaNo)
    Data["PrefiereRural"] = pd.Series(PrefiereRural_)
    Data["PrefiereCiudad"] = pd.Series(PrefiereCiudad)
    Data["PrefiereMaquinas"] = pd.Series(PrefiereMaquinas)
    Data["PrefierePersonas"] = pd.Series(PrefierePersonas)


    Data.pop("Genero")
    Data.pop("InteresEnTecnologia")
    Data.pop("PrefiereMaquinasOPersonas")





    for ind, NumCarreras in enumerate(Data.NumCarrerasEmpezada):

        a_row =Data.iloc[ind]
        if NumCarreras == "3 o más" and a_row.AntepenultimaSatisfaccion > 5:
            copy = a_row
            copy.UltimaSatisfaccion = a_row.AntepenultimaSatisfaccion
            copy.UltimaAcabado = a_row.AntepenultimaAcabado
            copy.UltimaCarrera = a_row.AntepenultimaCarrera
            copy.NumCarrerasEmpezada = 1;
            Data = Data.append(copy, ignore_index=True)


        if (NumCarreras == "3 o más" or NumCarreras == "2") and a_row.PenultimaSatisfaccion > 5:
            copy = a_row
            copy.UltimaSatisfaccion = a_row.PenultimaSatisfaccion
            copy.UltimaAcabado = a_row.PenultimaAcabado
            copy.UltimaCarrera = a_row.PenultimaCarrera
            copy.NumCarrerasEmpezada = 1;
            Data = Data.append(copy, ignore_index=True)




    Data = Data[Data["UltimaSatisfaccion"] >= 4]



    Data.pop("AntepenultimaSatisfaccion")
    Data.pop("AntepenultimaAcabado")
    Data.pop("AntepenultimaCarrera")
    Data.pop("PenultimaSatisfaccion")
    Data.pop("PenultimaAcabado")
    Data.pop("PenultimaCarrera")
    Data.pop("UltimaAcabado")
    Data.pop("UltimaSatisfaccion")
    Data.pop("NumCarrerasEmpezada")



    '''

    
    'Anho', 'AnhoCarrera', 'Nhermanos', 'HermanoMayor', 'Compras',
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

    columnas = ['AnhoCarrera','Anho', 'EstudiosRelaccionadosConLosPadres', 'EsMujer', 'EsNoBinario' ]

    for columna in columnas:
        Data.pop(columna)

    printColumnValues(Data)

    changeDtype(Data)

    return Data


def preprocessingInput(Data):




    Data.HermanoMayor = 1 if Data.HermanoMayor else 0

    Data["Nhermanos"] = 6 if Data["Nhermanos"] > 5 else Data["Nhermanos"]


    Data["InteresTecnologiaTalVez"] = 1 if Data['InteresEnTecnologia'] == "Un poco" else 0
    Data["InteresTecnologiaNo"] = 1 if Data['InteresEnTecnologia'] == "No" else 0

    Data.pop("InteresEnTecnologia")


    Data["PrefiereRural"] = 1 if Data['PrefiereRural'] == "Rural" else 0
    Data["PrefiereCiudad"] = 1 if Data['PrefiereRural'] == "Ciudad" else 0


    columnas = ['IrseDeEspanha', 'CuidarPersonas', 'EscucharPersonas',
                'CuidarAnimales', 'Sociable', 'Creativo', 'EfectoNegativoSangre']
    for columna in columnas:
        Data[columna] = 0 if Data[columna] == "No" else 1


    if Data.Organizada == "No es lo mío.":
        Data.Organizada = 0
    elif Data.Organizada == "Lo mínimo para alcanzar mis objetivos.":
        Data.Organizada = 1
    elif Data.Organizada == "Bastante, organizo mis asuntos personales.":
        Data.Organizada = 2
    elif Data.Organizada == "Mucho, me gusta también planificar asuntos ajenos o grupales.":
        Data.Organizada = 3



    Data["PrefierePersonas"] = 1 if Data['PrefiereMaquinasOPersonas'] == "Personas" else 0
    Data["PrefiereMaquinas"] = 1 if Data['PrefiereMaquinasOPersonas'] == "Máquinas" else 0

    Data.pop("PrefiereMaquinasOPersonas")

    DataDF = pd.DataFrame(data=Data)

    changeDtype(DataDF)


    return DataDF

def printColumnValues(Data):

    for column in Data:
        print(Data[column].value_counts())

def changeDtype(Data):

    for column in Data:
        if column != "UltimaCarrera":
            Data[column] = Data[column].astype(np.float32)