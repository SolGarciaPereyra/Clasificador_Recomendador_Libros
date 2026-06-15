import matplotlib.pyplot as plt
import pandas as pd

def plot_rating(df): #
    plt.hist(df["average_rating"], bins=50, edgecolor="white")
    plt.xlabel("Rating Promedio")
    plt.ylabel("Cantidad de libros")
    plt.title("Distribución de ratings")
    plt.show()

def plot_boxplot_rating(df):
    ratings = df["average_rating"]
    plt.boxplot(ratings,vert=True,patch_artist=True)

    plt.ylabel("Rating promedio")
    plt.title("Distribución de ratings")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stars(df): #
    porcentajes = (df["estrellas"].value_counts(normalize = True).sort_index()*100)
    bars = plt.bar(porcentajes.index, porcentajes.values)
    
    plt.xlabel("Estrellas")
    plt.ylabel("% de libros")
    plt.title("Distribución de estrellas")
    
    for bar in bars:
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, altura,f"{altura:.1f}%", ha="center", va="bottom")
    plt.show()

def plot_generos_mas_frecuentes(contador_generos): #
    contador_ordenado = contador_generos.sort_values()
    plt.figure(figsize=(10, 6))
    plt.barh(contador_ordenado.index, contador_ordenado.values)

    plt.title("Géneros más frecuentes")
    plt.xlabel("Frecuencia")
    plt.ylabel("Género")
    plt.tight_layout()
    plt.show()

def plot_rating_por_genero(df): #
    df_generos = df.explode("genres")
    promedio_rating = (df_generos.groupby("genres")["average_rating"].mean().sort_values())

    plt.figure(figsize=(10, 6))
    plt.barh(promedio_rating.index, promedio_rating.values)
    plt.xlabel("Average Rating")
    plt.ylabel("Género")
    plt.title("Rating promedio por género")
    plt.tight_layout()
    plt.show()

def plot_boxplot_rating_por_genero(df): #
    plt.figure(figsize=(12, 6))
    df_generos = df.explode("genres")
    df_generos.boxplot(column="average_rating",by="genres",rot=90)

    plt.xlabel("Género")
    plt.ylabel("Average Rating")
    plt.title("Distribución de ratings por género")
    plt.suptitle("")  # elimina el titulo automático de pandas
    plt.tight_layout()
    plt.show()

def plot_cantidad_generos_por_libro(df):
    cantidad_generos = df["genres"].apply(len)
    distribucion = (cantidad_generos.value_counts().sort_index())

    plt.figure(figsize=(10, 6))
    plt.bar(distribucion.index, distribucion.values)

    plt.xlabel("Cantidad de géneros")
    plt.ylabel("Cantidad de libros")
    plt.title("Distribución de géneros por libro")

    plt.tight_layout()
    plt.show()
