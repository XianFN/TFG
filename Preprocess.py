import tensorflow as tf
import numpy as np
import pandas as pd

def preprocessingDataset(Data):
    print("holahola")
    print(Data.columns)
    print(Data.shape)

    missing_values_count = Data.isnull().sum()

    # look at the # of missing points in the first ten columns
    print(missing_values_count[0:10])


    Data.pop("MarcaTemporal")
    Data.pop("ConsentimientoDatos")
    Data = Data.rename(columns={'Tecnologia.1': 'AsignaturaTecnologia'})


    for ind, year in enumerate(Data.Anho):
        Data.Anho[ind] = 2022 - Data.Anho[ind]
    print(Data.Anho)



    #print(Data.Genero.value_counts())
    EsMujer={}
    EsNoBinario = {}
    for ind, genero in enumerate(Data.Genero):

        if Data.Genero[ind] == "Mujer":
            EsMujer[ind] = 1
        else:
            EsMujer[ind] = 0

        if Data.Genero[ind] == "Prefiero no contestar" or Data.Genero[ind] == "GÃ©nero no binario":
            EsNoBinario[ind] = 1
        else:
            EsNoBinario[ind] = 0


    Data["EsMujer"] = pd.Series(EsMujer)
    Data["EsNoBinario"] = pd.Series(EsNoBinario)
    #print(Data.EsMujer)
    #print(Data.EsNoBinario)
    Data.pop("Genero")

    #print(Data.Nhermanos.value_counts())
    # COMPROBAR LOS NA TODO
    for ind, Nhermanos in enumerate(Data.Nhermanos):
        if Data.Nhermanos[ind] == "Prefiero no contestar":
            Data.Nhermanos[ind] = 0
        elif Data.Nhermanos[ind] == "+5":
            Data.Nhermanos[ind] = 6
    #print(Data.Nhermanos.value_counts())

    #print(Data.HermanoMayor.value_counts())

    #COMPROBAR LOS NA TODO

    for ind, hermanoMayor in enumerate(Data.HermanoMayor):
        if Data.HermanoMayor[ind] == "No":
            Data.HermanoMayor[ind] = 0
        else:
            Data.HermanoMayor[ind] = 1

    #print(Data.HermanoMayor.value_counts())

    #print(Data.Compras.value_counts())

    print(Data.columns)

    print(Data.InteresEnTecnologia.value_counts())

    InteresTecnologiaNo = {}
    InteresTecnologiaTalVez = {}
    for ind, interes in enumerate(Data.InteresEnTecnologia):
        InteresTecnologiaNo[ind] = 0
        InteresTecnologiaTalVez[ind] = 0
        if Data.InteresEnTecnologia[ind] == "Un poco":
            InteresTecnologiaNo[ind] = 0
            InteresTecnologiaTalVez[ind] = 1
        elif Data.InteresEnTecnologia[ind] == "No":
            InteresTecnologiaNo[ind] = 1
            InteresTecnologiaTalVez[ind] = 0

    Data["InteresTecnologiaTalVez"] = pd.Series(InteresTecnologiaTalVez)
    Data["InteresTecnologiaNo"] = pd.Series(InteresTecnologiaNo)
    Data.pop("InteresEnTecnologia")

    print(Data.InteresTecnologiaTalVez.value_counts())
    print(Data.InteresTecnologiaNo.value_counts())

    asignaturas = ['Matematicas', 'LenguaLiteratura', 'Ingles','Historia', 'EducacionFisica', 'Fisica', 'Quimica', 'DibujoTecnico','AsignaturaTecnologia', 'Filosofia', 'Biologia', 'LatinGriego', 'Frances', 'Religion']

    #'''QUITAR
    print(Data.Matematicas.value_counts())

    for asignatura in asignaturas:
        ind = 0
        for ind, persona in enumerate(Data[asignatura]):
            if persona == "No la cursÃ©":
                Data[asignatura][ind] = 0
        #print(Data[asignatura].value_counts())


    #'''

    print(Data.PrefiereRural.value_counts())

    PrefiereRural_ = {}
    PrefiereCiudad = {}
    for ind, preferencia in enumerate(Data.PrefiereRural):
        PrefiereRural_[ind] = 0
        PrefiereCiudad[ind] = 0
        if preferencia == "Rural":
            PrefiereCiudad[ind] = 0
            PrefiereRural_[ind] = 1
        elif preferencia == "Ciudad":
            PrefiereCiudad[ind] = 1
            PrefiereRural_[ind] = 0

    Data["PrefiereRural"] = pd.Series(PrefiereRural_)
    Data["PrefiereCiudad"] = pd.Series(PrefiereCiudad)

    # COMPROBAR que todas las columans se pueda usar así los NA TODO

    columnas=['EstudiosRelaccionadosConLosPadres', 'IrseDeEspanha','CuidarPersonas', 'EscucharPersonas', 'CuidarAnimales', 'Sociable', 'Creativo', 'EfectoNegativoSangre' ]
    for columna in columnas:
        ind = 0
        for ind, persona in enumerate(Data[columna]):
            if persona == "No":
                Data[columna][ind] = 0
            else:
                Data[columna][ind] = 1


    for ind, persona in enumerate(Data.Organizada):
        if persona == "No es lo mÃ­o.":
            Data.Organizada[ind] = 0
        elif persona == "Lo mÃ­nimo para alcanzar mis objetivos.":
            Data.Organizada[ind] = 1
        elif persona == "Bastante, organizo mis asuntos personales.":
            Data.Organizada[ind] = 2
        elif persona == "Mucho, me gusta tambiÃ©n planificar asuntos ajenos o grupales.":
            Data.Organizada[ind] = 3

    PrefiereMaquinas = {}
    PrefierePersonas = {}
    for ind, persona in enumerate(Data.PrefiereMaquinasOPersonas):
        PrefiereMaquinas[ind] = 0
        PrefierePersonas[ind] = 0
        if persona == "Personas":
            PrefierePersonas[ind] = 1
        elif persona == "MÃ¡quinas":
            PrefiereMaquinas[ind] = 1


    Data["PrefiereMaquinas"] = pd.Series(PrefiereMaquinas)
    Data["PrefierePersonas"] = pd.Series(PrefierePersonas)
    Data.pop("PrefiereMaquinasOPersonas")



    print(Data.shape)


    for ind, NumCarreras in enumerate(Data.NumCarrerasEmpezada):

        a_row =Data.iloc[ind]
        if NumCarreras == "3 o mÃ¡s" and a_row.AntepenultimaSatisfaccion > 5:
            copy = a_row
            copy.UltimaSatisfaccion = a_row.AntepenultimaSatisfaccion
            copy.UltimaAcabado = a_row.AntepenultimaAcabado
            copy.UltimaCarrera = a_row.AntepenultimaCarrera
            copy.NumCarrerasEmpezada = 1;
            Data = Data.append(copy, ignore_index=True)


        if (NumCarreras == "3 o mÃ¡s" or NumCarreras == "2") and a_row.PenultimaSatisfaccion > 5:
            copy = a_row
            copy.UltimaSatisfaccion = a_row.PenultimaSatisfaccion
            copy.UltimaAcabado = a_row.PenultimaAcabado
            copy.UltimaCarrera = a_row.PenultimaCarrera
            copy.NumCarrerasEmpezada = 1;
            Data = Data.append(copy, ignore_index=True)

    print(Data.shape)

    #TODO Borrar todas las filas, que la satisfaccion sea menor que 5

    #printColumnValues(Data)


    Data.pop("AntepenultimaSatisfaccion")
    Data.pop("AntepenultimaAcabado")
    Data.pop("AntepenultimaCarrera")
    Data.pop("PenultimaSatisfaccion")
    Data.pop("PenultimaAcabado")
    Data.pop("PenultimaCarrera")
    Data.pop("UltimaAcabado")
    Data.pop("UltimaSatisfaccion")
    Data.pop("NumCarrerasEmpezada")

    changeDtype(Data)
    #print(Data.dtypes)
    # Empeora con 'Compras',
    #        'EscapeRooms', 'Animales', 'Coches'
    '''
    missing_values_count = Data.isnull().sum()
    # look at the # of missing points in the first ten columns
    print(missing_values_count[20:30])
    
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
    print(Data.columns)
    columnas = ['AnhoCarrera','Anho', 'EstudiosRelaccionadosConLosPadres', 'EsMujer', 'EsNoBinario' ]

    for columna in columnas:
        Data.pop(columna)


    return Data


def preprocessingInput(Data):


    print(Data.shape)

    #  missing_values_count = Data.isnull().sum()

    # look at the # of missing points in the first ten columns
    #     print(missing_values_count[0:10])



    # COMPROBAR LOS NA TODO
    print(Data)



    Data.HermanoMayor = 1 if Data.HermanoMayor else 0



    Data["InteresTecnologiaTalVez"] = 1 if Data['InteresEnTecnologia'] == "Un poco" else 0
    Data["InteresTecnologiaNo"] = 1 if Data['InteresEnTecnologia'] == "No" else 0

    Data.pop("InteresEnTecnologia")

    print(Data.InteresTecnologiaTalVez)
    print(Data.InteresTecnologiaNo)


    print(Data.PrefiereRural)

    Data["PrefiereRural"] = 1 if Data['PrefiereRural'] == "Rural" else 0
    Data["PrefiereCiudad"] = 1 if Data['PrefiereRural'] == "Ciudad" else 0



    # COMPROBAR que todas las columans se pueda usar así los NA TODO

    columnas = ['IrseDeEspanha', 'CuidarPersonas', 'EscucharPersonas',
                'CuidarAnimales', 'Sociable', 'Creativo', 'EfectoNegativoSangre']
    for columna in columnas:
        Data[columna] = 0 if Data[columna] == "No" else 1


    print(Data.Organizada)

    if Data.Organizada == "No es lo mío.":
        Data.Organizada = 0
    elif Data.Organizada == "Lo mínimo para alcanzar mis objetivos.":
        Data.Organizada = 1
    elif Data.Organizada == "Bastante, organizo mis asuntos personales.":
        Data.Organizada = 2
    elif Data.Organizada == "Mucho, me gusta también planificar asuntos ajenos o grupales.":
        Data.Organizada = 3

    print(Data.Organizada)

    Data["PrefierePersonas"] =  1 if Data['PrefiereMaquinasOPersonas'] == "Personas" else 0
    Data["PrefiereMaquinas"] = 1 if Data['PrefiereMaquinasOPersonas'] == "Máquinas" else 0

    Data.pop("PrefiereMaquinasOPersonas")

    print(Data.shape)

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