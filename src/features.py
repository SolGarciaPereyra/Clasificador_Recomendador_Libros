import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from langdetect import detect, DetectorFactory
from sklearn.decomposition import IncrementalPCA

def limpiar_features_no_necesarias(df, lista_features = ["is_series", "is_ebook", "publication_year", "num_pages", 'format_AUDIO', 'format_COMIC', 'format_EBOOK', 'format_HARDCOVER', 'format_OTHER', 'format_PAPERBACK', 'format_UNKNOWN']):
    df_entrenamiento = df[lista_features] 
    return df_entrenamiento

def es_parte_de_saga(series):
    #la feature series contiene una lista vacia si es un standalone, o un codigo si es parte de una saga
    return len(series) > 0

def multi_binarizer_format(df):
    #misma logica que one hot pero permite varias features en 1
    mlb = MultiLabelBinarizer()
    format_encoded = mlb.fit_transform(df["format_group"])
    format_df = pd.DataFrame(format_encoded, columns = mlb.classes_, index = df.index)
    df = df.join(format_df)
    return df

def one_hot_language(df):
    #language: una vez habiendo simplificado e imputado los language_code, aplicamos one hot encoding
    encoder_format = OneHotEncoder(sparse_output=False)
    encoded_format = encoder_format.fit_transform(df[["language_code"]])
    encoded_df = pd.DataFrame(encoded_format, columns = encoder_format.get_feature_names_out(["language_code"]), index = df.index)
    df = df.join(encoded_df)
    return df

def one_hot_format(df): 
    #one hot encoding de la feature format
    encoder_format = OneHotEncoder(sparse_output=False)
    encoded_format = encoder_format.fit_transform(df[["format_group"]])
    encoded_df = pd.DataFrame(encoded_format, columns = encoder_format.get_feature_names_out(["format_group"]), index = df.index)
    df = df.join(encoded_df)
    return df

def multi_binarizer_genre(df):
    #misma logica que one hot pero permite varias features en 1
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df["genres"])
    genres_df = pd.DataFrame(genres_encoded, columns = mlb.classes_, index = df.index)
    df = df.join(genres_df)
    return df

def transformar_target_a_estrellas(target):
    #transforma el rating continuo en una distribucion discreta de estrellas
    return target.round().astype(int)


############################
# Feature format
############################

def normalize_format(value):
    """Unifica formatos repetidos bajo diferentes nombres y los asigna"""
    #Si el valor no es un string:
    if not isinstance(value, str):
        return "UNKNOWN"
    #Si no hay un valor para la muestra o es un espacio en blanco:
    s = value.strip().lower()
    if s == "":
        return "UNKNOWN"
    if "unknown binding" in s or s == "unknown":
        return "UNKNOWN"

    #Reemplazamos &amp; (código HTML para &) por solo &:
    s = s.replace("&amp;", "&")

    #Hacemos unas correcciones de cosas que aparecen frecuentemente en el dataset:
    s = s.replace("papperback", "paperback")
    s = s.replace("papaerback", "paperback")
    s = s.replace("paberback", "paperback")
    s = s.replace("hardcovver", "hardcover")
    s = s.replace("hardover", "hardcover")
    s = s.replace("harcover", "hardcover")


    #Ebook (digital)
    ebook_terms = [
        "ebook", "e-book", "e book", "kindle", "nook", "nookbook", "ibook", "epub", "pdf", "electronic",
        "digital", "online", "wattpad", "webtoon", "webcomic", "web comic", "online fiction", "online fanfiction",
        "free online fiction", "free online read", "online read", "online serial", "online publication", "digital comic",
        "fanfiction"
    ]

    #Formato de audios
    audio_terms = [
        "audio", "audiobook", "audible", "audio cd", "audio cassette", "audio book",
        "audio play", "audio lecture", "audio podcast", "mp3", "cd audio", "cassette",
        "playaway", "digital audio", "overdrive audio"
    ]

    #Hardcover (hardback)
    hardcover_terms = [
        "hardcover", "hard cover", "hardback", "hardbound", "hard bound", "casebound", "library binding",
        "school binding", "cloth bound", "clothbound", "leather bound", "leatherbound", "slipcased",
        "slipcase", "flexbound", "hardcover with slipcase"
    ]

    #Paperback (softcover)
    paperback_terms = [
        "paperback", "softcover", "soft cover", "trade paperback", "mass market paperback", "paper back",
        "paperbound", "pocket paperback", "capa mole", "brochura", "broche", "brochure", "brossura", "paperback with dvd",
        "paperback with flaps", "paperback (large print)", "perfect paperback", "mass paperback", "softback",
        "quality paperback", "paperback manga", "spiral-bound", "spiral bound", "chapbook", "paper", 
    ]

    #Comic / manga / graphic
    comic_terms = ["comic", "graphic novel", "manga", "comic book", "graphic", "manhwa", "manhua", "fancomic"]

    if any(term in s for term in audio_terms):
        return "AUDIO"

    if any(term in s for term in comic_terms):
        return "COMIC"

    if any(term in s for term in hardcover_terms):
        return "HARDCOVER"

    if any(term in s for term in paperback_terms):
        return "PAPERBACK"

    if any(term in s for term in ebook_terms):
        return "EBOOK"

    return "OTHER"


############################
# Feature genero
############################

#Defino los terminos que permiten identificar cada genero
generos_compuestos = ["science-fiction", "young-adult", "non-fiction", "sci-fi", "self-help"] #generos que no hay que separar

fiction_terms = ["fiction", "novel", "novels", "narrativa", "lit", "literary", "literature"]
non_fiction_terms = ["non-fiction", "nonfiction", "nonfic", "nf"]
fantasy_terms = ["fantasy", "magic", "magical", "witches", "dragons", "fae", "high"]
science_fiction_terms = ["science-fiction", "sci-fi", "scifi", "sf", "sciencefiction", "science fiction", "futuristic", "space", "aliens"]
mystery_terms = ["mystery", "mysteries", "detective", "detectives", "cozy", "cozies"]
thriller_terms = ["thriller", "thrillers", "suspense", "psychological"]
romance_terms = ["romance", "romances", "romantic", "chick", "lovers"]
horror_terms = ["horror", "scary", "creepy", "gothic"]
historical_terms = ["historical", "victorian", "medieval", "regency"]
young_adult_terms = ["ya", "young-adult", "young adult", "teen", "teens","tween"]
children_terms = ["children", "childrens", "kids", "picture", "picturebooks", "kindergarten", "preschool", "elementary"]
adventure_terms = ["adventure", "action", "survival"]
contemporary_terms = ["contemporary", "modern", "realistic"]
dystopian_terms = ["dystopian", "dystopia", "apocalyptic", "apocalypse", "postapocalyptic"]
paranormal_terms = ["paranormal", "supernatural"]
urban_terms = ["urban", "urban fantasy"] #
crime_terms = ["crime", "truecrime", "police"]
biography_terms = ["biography", "biographies", "bio", "biographical"]
memoir_terms = ["memoir", "memoirs", "autobiography", "autobiographical"]
self_help_terms = ["self-help", "self help", "selfhelp", "selfimprovement", "selfdevelopment", "personaldevelopment", "personalgrowth", "leadership"]
history_terms = ["history", "historical", "historial"]
psychology_terms = ["psychology", "mentalhealth, mentalillness", "mental", "health", "illness", "mental health", "metal illness"]
philosophy_terms = ["philosophy", "philosophical"]
poetry_terms = ["poetry", "poems", "poem", "verse"]
classics_terms = ["classics", "classic", "classical", "literary"]
graphic_novel_terms = ["graphic-novel", "graphicnovel", "graphic", "graphic novel", "comics", "comic", "manga", "mangas"]
short_stories_terms = ["short stories", "short", "shortstories", "shorts", "shortstory", "anthology", "amthologies", "novella", "novellas"]
humor_terms = ["humor", "humour", "humorous", "funny", "comedy", "satire"]
religion_terms = ["religion", "religious", "christian", "christianity", "theology", "spiritual", "spirituality", "faith"]

MAPEO_GENEROS = {"fiction": fiction_terms, "non fiction": non_fiction_terms, "fantasy": fantasy_terms, "science fiction": science_fiction_terms,
"mystery": mystery_terms, "thriller": thriller_terms, "romance": romance_terms, "horror": horror_terms, "historical": historical_terms, 
"young adult": young_adult_terms, "children": children_terms, "adventure": adventure_terms, "contemporary": contemporary_terms,
"dystopian": dystopian_terms, "paranormal": paranormal_terms, "urban": urban_terms, "crime": crime_terms, "biography": biography_terms, 
"memoir": memoir_terms, "self help": self_help_terms, "history": history_terms, "psychology": psychology_terms, "poetry": poetry_terms, 
"classics": classics_terms, "graphic novel": graphic_novel_terms, "short stories": short_stories_terms, "humor": humor_terms, "religion": religion_terms}

def obtener_score(item):
    genero, score = item
    return score

def extraer_generos_libro(shelves):
    """Determina los generos de un libro a partir de su diccionario de shelves mas repetidos"""
    generos = {}

    for shelf in shelves:
        nombre = shelf["name"].lower()
        count = int(shelf["count"])

        #separo los generos con guion en el medio, a menos que esten en la lista de los que se permiten
        if nombre in generos_compuestos:
            palabras = [nombre]
        else:
            palabras = nombre.split("-")

        #verifico si las palabras corresponden con algun genero definido en las listas de mas arriba
        for genero, terminos in MAPEO_GENEROS.items():
            for termino in terminos:
                if termino in palabras:
                    generos[genero] = generos.get(genero, 0) + count
    
    generos_ordenados = sorted(generos.items(), key = obtener_score, reverse=True)
    top6 = []
    for genero, score in generos_ordenados[:6]:
        top6.append(genero)
    return top6


############################
# Feature author
############################

def primer_tiene_role_vacio(authors_list):
    return authors_list[0]["role"] == ""

def extraer_autor_principal(autores_sin_preprocesar):
    """
    autores_sin preprocesar: [{authorId, role}, {authorid, role})]
    Analizamos el diccionario para quedarnos unicamente con el escritor
    Asumimos que el primer elemento de la lista es el autor principal (en sentido de escritor), es el orden usual de Goodreads
    """
    #buscar el primer elemento con role vacio
    for dic_autores in autores_sin_preprocesar:
        if dic_autores["role"] == "":
            return dic_autores["author_id"]
    return autores_sin_preprocesar[0]["author_id"]

def crear_codificador_categorico(X_entrenamiento, y_entrenamiento, columnas_categoricas, tipo_target, smoothing=20):
    #Aca vamos a guardar toda la info de la codificación
    codificador = {}
    #Calculamos la media global de la variable objetivo (util para usarlo como valor de respaldo para posibles 
    #categorías que aparezcan en val o test, pero no hayan estado en train)
    #Imputariamos con la media y no con la mediana porque, al trabajar con target encoding, 
    #queremos estimar E[target | publisher], y su mejor estimacion es la media.
    # media_global_target = y_entrenamiento.mean()

    #Recorremos cada columna categórica que queremos codificar:
    for columna in columnas_categoricas:
        datos_temporales = pd.DataFrame({columna: X_entrenamiento[columna], tipo_target: y_entrenamiento})
        #Hacemos la codifcacion de frecuencia (ej: cantidad de libros por publisher)
        mapa_frecuencia = datos_temporales[columna].value_counts().to_dict()
        codificador[columna] = {"frecuencia": mapa_frecuencia}

    return codificador

def transformar_features_categoricas(X_datos, codificador):
    X_transformado = X_datos.copy()

    #Recorremos cada columna que fue codificada:
    for columna in codificador.keys():
        X_transformado[f"{columna}_frecuencia"] = (X_transformado[columna].map(codificador[columna]["frecuencia"]).fillna(0))

    return X_transformado

def calcular_diferencia_anios(publication_year):
    return publication_year.max() - publication_year.min()

def calcular_antiguedad_autor(df):
    """Calcula la diferencia entre el libro mas reciente publicado y el mas antiguo"""
    df["author_antiguedad"] = (df.groupby("author_id_principal")["publication_year"].transform(calcular_diferencia_anios))
    return df


############################
# Feature language
############################

def detectar_language_code(fila):
    try:
        descripcion = str(fila["description"]).strip()
        #Priorizarmos detectar el idioma usando la feature "description" si es que tiene suficiente texto:
        if ((pd.notna(descripcion)) and (len(descripcion) > 50)):
            return detect(descripcion)
        
        #Pero, si la muestra no tiene descripción, usamos el titulo:
        title = str(fila["title"]).strip()
        if len(title) > 0:
            return detect(title)

        #Es el caso en el que la muestra no tiene ni titulo ni descripcion lo suficientemente larga:
        return "unknown"

    #Entramos si salta algun error en la seccion "try" (ej: no existe alguna de las dos columans mencioandas, o las funciones
    #usadas tiran algun error)
    except:
        return "unknown"


