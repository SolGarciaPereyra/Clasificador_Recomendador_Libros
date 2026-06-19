import pandas as pd
import numpy as np
from src.features import limpiar_features_no_necesarias

def df_target_to_numpy(target, dfs):
    #target puede ser "average_rating" para regresion, o "estrellas" para clasificacion
    ys = []
    for df in dfs:
        ys.append(df[target].to_numpy())
    return ys

def df_features_to_numpy(features, dfs):
    #[df_train, df_val, df_test]
    Xs = []
    for df in dfs:
        df_limpiado = limpiar_features_no_necesarias(df, features)
        Xs.append(df_limpiado.to_numpy())
    return Xs

def fusionar_libros(grupo):
    #Comenzamos por ordenar (del mas antiguo al mas reciente) las ediciones del libro por año de publicacion
    #Vamos a usar a la primera fila (la más antigua) como "base" para completar ciertas features
    grupo = grupo.sort_values("publication_year")
    #Seleccionamos la edicion mas antigua:
    fila_base = grupo.iloc[0]

    #Calculamos el rating promedio ponderado: 
    ratings_count_total = grupo["ratings_count"].sum()
    if (ratings_count_total > 0):
        average_rating = np.average(grupo["average_rating"], weights=grupo["ratings_count"])
    else: #Caso muy raro que suceda, pero por las dudas...
        average_rating = grupo["average_rating"].mean()

    #Agrupamos todos los formatos unicos en los que existe el libro
    formatos = sorted(set(grupo["format_group"].dropna()))
    #Como cada muestra tiene su propia lista de generos, los agrupamos a todos para formar una sola lista de generos unicos
    generos = sorted(set(genero for lista in grupo["genres"] for genero in lista))
    #Elegimos la descripcion mas larga:
    descripcion = max(grupo["description"].fillna(""), key = len)
    #Construimos la muestra fusionada:
        #"title", "authors", "publisher", "num_pages", "language_code", "title_text" --> se toman de la edicion base (la mas antigua)
        #"publication_year" --> el mas antiguo entre todas las ediciones
        #"average_rating" --> lo calculamos como el promedio ponderado
        #"ratings_count" --> total de ratings entre todas las ediciones
        #"is_series" --> es True si y solo si al menos una de las ediciones tiene indicado que es una saga
    libro_fusionado = pd.Series({"title": fila_base["title"], "authors": fila_base["authors"], "publisher": fila_base["publisher"], "publication_year": grupo["publication_year"].min(), "num_pages": fila_base["num_pages"], "language_code": fila_base["language_code"], "average_rating": average_rating,
                            "ratings_count": ratings_count_total, "genres": generos, "format_group": formatos, "description": descripcion, "title_text": fila_base["title"], "is_series": grupo["is_series"].any()})
    return libro_fusionado
