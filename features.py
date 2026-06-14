import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def limpiar_features_no_necesarias(df):
    df_entrenamiento = df[["is_series", "is_ebook", "publication_year", "num_pages", 'format_AUDIO', 'format_COMIC', 'format_EBOOK', 'format_HARDCOVER', 'format_OTHER', 'format_PAPERBACK', 'format_UNKNOWN']] 
    return df_entrenamiento

def es_parte_de_saga(series):
    #la feature series contiene una lista vacia si es un standalone, o un codigo si es parte de una saga
    return len(series) > 0

def transformar_features(df):
    #is_series: feature binaria, true si es parte de una saga
    df["is_series"] = df["series"].apply(es_parte_de_saga)
    df = df.drop(columns = ["series"])
    #is_ebook: si vemos que es importante, antes eliminar los datos que estan vacios #TODO
    df["is_ebook"] = df["is_ebook"] == "true"
    #format: una vez habiendo simplificado los formatos, aplicamos one hot encoding
    df["format"] = df["format"].apply(normalize_format)
    encoder_format = OneHotEncoder(sparse_output=False)
    encoded_format = encoder_format.fit_transform(df[['format']])
    encoded_df = pd.DataFrame(encoded_format, columns = encoder_format.get_feature_names_out())
    df = df.join(encoded_df)
    #publication_year, num_pages: transformar a numero (estaban en formato string) #TODO: eliminar o ver como imputar nan
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors='coerce')
    df["num_pages"] = pd.to_numeric(df["num_pages"], errors='coerce')
    return df

def transformar_target_a_estrellas(df_target):
    df_target["stars"] = df_target["average_rating"].round().astype(int)
    return df_target["stars"]

#SACADO DEL EDA:
def normalize_format(value):
    if not isinstance(value, str):
        return "UNKNOWN"
    s = value.strip().lower()
    if s == "":
        return "UNKNOWN"

    # Ebook / digital
    ebook_terms = [
        "ebook", "e-book", "kindle", "nook", "ibook", "epub", "pdf",
        "online fiction", "online free", "online read", "online", "digital",
        "e book", "e-book", "ebook and paperback", "ebook, paperback",
        "nook ebook", "e-book .pdf", "e book, paperback", "e-pub", "ebook pdf"
    ]

    # Audio formats
    audio_terms = [
        "audio", "audiobook", "audio cd", "mp3", "cassette", "audible",
        "digital audio", "audio lecture", "audio book", "audiobook, audible",
        "audio cassette", "mp3 audio", "mp3 book"
    ]

    # Hardcover / hardback
    hardcover_terms = [
        "hardcover", "hard cover", "hardback", "hardbound", "slipcased",
        "hardcover first edition", "hardcover comic", "hardcover (large print)",
        "faux leather", "hardcover with slipcase", "hardcover art"
    ]

    # Paperback / softcover
    paperback_terms = [
        "paperback", "softcover", "soft cover", "trade paperback",
        "mass market paperback", "paper back", "paperbound", "pocket paperback",
        "capa mole", "paper", "brochura", "broche", "book", "brochure",
        "brossura", "paperback with dvd", "paperback with flaps",
        "paperback (large print)", "perfect paperback", "paperback manga"
    ]

    # Comic / manga / graphic
    comic_terms = [
        "comic", "graphic novel", "manga", "webcomic", "fancomic", "comic book",
        "digital comic", "comic book", "graphic"
    ]

    s_norm = s.replace("&amp;", "&")

    if any(term in s_norm for term in ebook_terms) and not any(term in s_norm for term in audio_terms):
        return "EBOOK"
    if any(term in s_norm for term in audio_terms):
        return "AUDIO"
    if any(term in s_norm for term in hardcover_terms):
        return "HARDCOVER"
    if any(term in s_norm for term in paperback_terms):
        return "PAPERBACK"
    if any(term in s_norm for term in comic_terms):
        return "COMIC"

    return "OTHER"

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
- publication_year: lo dejamos asi
- url: no se si nos sirve, tal vez para despues. diria que volar
- image_url: volar
- book_id: ni idea
- ratings count: dejar asi
- work id: volar
- titulo: nos lo quedamos asi pero no lo usamos para el entrenamiento
- title_without_series: volar, nos quedamos solo con el titulo
- authors: es un diccionario: como lo tratamos
- publisher: transformar a bool de si tiene publisher o no?
- num_pages: nos lo quedamos asi
- publication_day: volar
- isbn13: volar
"""