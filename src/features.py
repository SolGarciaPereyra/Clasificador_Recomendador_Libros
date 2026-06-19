import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from langdetect import detect, DetectorFactory

def limpiar_features_no_necesarias(df, lista_features = ["is_series", "is_ebook", "publication_year", "num_pages", 'format_AUDIO', 'format_COMIC', 'format_EBOOK', 'format_HARDCOVER', 'format_OTHER', 'format_PAPERBACK', 'format_UNKNOWN']):
    df_entrenamiento = df[lista_features] 
    return df_entrenamiento

def es_parte_de_saga(series):
    #la feature series contiene una lista vacia si es un standalone, o un codigo si es parte de una saga
    return len(series) > 0

def one_hot_format(df):
    #format: una vez habiendo simplificado los formatos, aplicamos one hot encoding
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
    return target.round().astype(int)

#SACADO DEL EDA:
def normalize_format(value):
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

#ANALISIS DE GENEROS
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
urban_terms = ["urban", "urban fantasy"] #urban fantasy
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

"""features que podemos volar
- isbn: podemos eliminar? o nos sirve para el mini recomendador
- text_reviews count: ok, pero no se usa en el entrenamiento
- series: transformar a is_standalone -> booleano
- country_code: hacer one hot encoding sobre si esta en US o no
- language_code: 
- popular_shelves: 
- asin: volar 
- is_ebook: graficar y ver si esto tiene impacto o no
- average_rating: target
- kindle_asin: volar
- similar_books: es hacer trampa?
- description: ver como tratarla
- format: transformarlo con one hot encoding
- publication_month: volar
- publication_year: lo dejamos asi --> eliminar outliers y, tambien los datos erroneos? --> que hacemos con sus NaN??
- url: no se si nos sirve, tal vez para despues. diria que volar
- image_url: volar
- book_id: ni idea
- ratings count: dejar asi --> eliminar...??
- work id: volar
- titulo: nos lo quedamos asi pero no lo usamos para el entrenamiento
- title_without_series: volar, nos quedamos solo con el titulo
- authors: es un diccionario: como lo tratamos
- publisher: transformar a bool de si tiene publisher o no?
- num_pages: nos lo quedamos asi --> eliminar outliers y valores ilogicos --> que hacemos con sus NaN??
- publication_day: volar
- isbn13: volar
"""