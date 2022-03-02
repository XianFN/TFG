import tensorflow as tf
import numpy as np
import pandas as pd

def hola(Data):
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











    printColumnValues(Data)






















    return Data

def printColumnValues(Data):

    for column in Data:
        print(Data[column].value_counts())
