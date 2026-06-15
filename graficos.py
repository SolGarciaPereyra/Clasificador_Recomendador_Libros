import seaborn as sns
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

def plot_boxplot_rating_por_formato(df_us_en):
    plt.figure(figsize = (6, 6))
    sns.boxplot(data = df_us_en, x = "format_group", y = "average_rating")
    plt.xlabel("Format group")
    plt.ylabel("Average rating")
    plt.title("average_rating vs format_group", fontweight = "bold")
    plt.xticks(rotation = 45)
    plt.show()

def plot_boxplot_rating_por_is_series(df_us_en):
    plt.figure(figsize = (6, 6))
    sns.boxplot(data  = df_us_en, x = "is_series", y = "average_rating")
    plt.xlabel("Es parte de una saga")
    plt.ylabel("Average rating")
    plt.title("average_rating vs is_series", fontweight = "bold")
    plt.show()

def plot_scatte_rating_por_num_pages(df_us_en):
    plt.figure(figsize = (6, 6))
    sns.scatterplot(data = df_us_en, x = "num_pages", y = "average_rating", alpha = 0.6)
    plt.xlabel("Numero de paginas")
    plt.ylabel("Rating promedio")
    plt.title("num_pages vs average_rating", fontweight = "bold")
    plt.show()

def plot_scatte_rating_por_publication_year(df_us_en):
    plt.figure(figsize = (6, 6))
    sns.scatterplot(data = df_us_en, x = "publication_year", y = "average_rating", alpha = 0.6)
    plt.xlabel("Año de publicacion")
    plt.ylabel("Rating promedio")
    plt.title("publication_year vs average_rating", fontweight = "bold")
    plt.show()

def plot_boxplot_rating_por_autor_frecuent(df_top):
    plt.figure(figsize = (10, 6))
    sns.boxplot(data = df_top, x = "author_id", y = "average_rating")
    plt.xlabel("Autor")
    plt.ylabel("Average rating")
    plt.title("Distribución de ratings en los autores con más libros", fontweight = "bold")
    plt.xticks(rotation = 45)
    plt.show()

