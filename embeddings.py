import pandas as pd
import torch
import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer


def entrenar_pca_para_texto(textos_variables, modelo, componentes_conservar = 50, batch_size = 2000, encode_batch_size = 256):
    #Convertimos "textos_variables" a una lista de strings, reemplazando los NaN por un string vacio
    textos_variables = textos_variables.fillna("").astype(str).tolist()
    #Decidimos usar el PCA incremental proque nos permite procesar los datos en bloques, sin cargar 
    #todo en memoria (util por nuestra cantidad de muestras, asi no llena la RAM)
    incremental_pca = IncrementalPCA(n_components = componentes_conservar)

    #Ahora, tenemos que hacer un mini entrenamiento del PCA, ajustandolo mediante batches de datos
    for num in range(0, len(textos_variables), batch_size):
        batch = textos_variables[num: num + batch_size]
        #generamos los mebeddings para este batch, usando el modelo pre-entrenado:
        embedding = modelo.encode(batch, convert_to_numpy = True, show_progress_bar = True, batch_size = encode_batch_size)
        #Ajustamos el PCA:
        incremental_pca.partial_fit(embedding)

    return incremental_pca

def generar_embeddings_pca(textos_variables, modelo, incremental_pca, batch_size = 1000):
    #Convertimos "textos_variables" a una lista de strings, reemplazando los NaN por un string vacio
    textos_variables = textos_variables.fillna("").astype(str).tolist()

    #Ahora si reducimos todos los datos:
    embeddings_reducidos = []
    for pos in range(0, len(textos_variables), batch_size):
        batch = textos_variables[pos : pos + batch_size]
        embedding = modelo.encode(batch, convert_to_numpy = True, show_progress_bar = False)
        #Aplicamos la transformacion PCA que entrenamos antes
        embedding_reducido = incremental_pca.transform(embedding)
        embeddings_reducidos.append(embedding_reducido)

    #Unimos todos los batches reducidos en un solo array
    embeddings_reducidos = np.vstack(embeddings_reducidos)
    return embeddings_reducidos

def crear_codificador_categorico(X_entrenamiento, y_entrenamiento, columnas_categoricas, tipo_target, smoothing = 20):
    #Aca vamos a guardar toda la info de la codificación
    codificador = {}
    #Calculamos la media global de la variable objetivo (util para usarlo como valor de respaldo para posibles 
    #categorías que aparezcan en val o test, pero no hayan estado en train)
    #Imputariamos con la media y no con la mediana porque, al trabajar con target encoding, 
    #queremos estimar E[target | author], y su mejor estimacion es la media.
    media_global_target = y_entrenamiento.mean()
    
    #Recorremos cada columna categórica que queremos codificar:
    for columna in columnas_categoricas:
        datos_temporales = pd.DataFrame({columna: X_entrenamiento[columna], f"{tipo_target}": y_entrenamiento})
        #Primero hacemos la codifcacion por media del target (ej: promedio del rating de la autora)
        estadisticas = (datos_temporales.groupby(columna)[f"{tipo_target}"].agg(["mean", "count"]))
        media_suavizada = (estadisticas["count"] * estadisticas["mean"] + smoothing * media_global_target) / (estadisticas["count"] + smoothing)
        mapa_media = media_suavizada.to_dict()
        #Ahora, hacemos la codifcacion de frecuencia (ej: cantidad de libros por publisher)
        mapa_frecuencia = datos_temporales[columna].value_counts().to_dict()
        codificador[columna] = {"media": mapa_media, "frecuencia": mapa_frecuencia, "media_global": media_global_target}
    
    return codificador

#LEO DATASETS
chunks_train = pd.read_json("set_train_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_train:
    chunk_list.append(chunk)
df_train = pd.concat(chunk_list, ignore_index=True)
print(df_train.shape)

chunks_val = pd.read_json("set_val_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_val:
    chunk_list.append(chunk)
df_val = pd.concat(chunk_list, ignore_index=True)
print(df_val.shape)

chunks_test = pd.read_json("set_test_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_test:
    chunk_list.append(chunk)
df_test = pd.concat(chunk_list, ignore_index=True)
print(df_test.shape)

#test
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device}")

#embeddings descripcion
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("CUDA disponible:", torch.cuda.is_available())
print("Device:", modelo_embeddings.device)

descripcion_pca = entrenar_pca_para_texto(df_train["description"], modelo_embeddings, componentes_conservar = 50)
print("termino de entrenar")
print("empiezo guardar")
joblib.dump(descripcion_pca, "descripcion_pca.joblib")

#Set de entrenamiento:
descripcion_emb_train = generar_embeddings_pca(df_train["description"], modelo_embeddings, descripcion_pca)
print("termino embedding train")
np.save("descripcion_emb_train.npy", descripcion_emb_train)
print("temrino de guardar")
#Set de validacion:
descripcion_emb_val = generar_embeddings_pca(df_val["description"], modelo_embeddings, descripcion_pca)
print("termino embedding val")
np.save("descripcion_emb_val.npy", descripcion_emb_val)
print("termino guardado")
#Set de testeo:
descripcion_emb_test = generar_embeddings_pca(df_test["description"], modelo_embeddings, descripcion_pca)
print("termino embedding test")
np.save("descripcion_emb_test.npy", descripcion_emb_test)
print("termino de guardar")

#Entrenamos el PCA solo con el set de train
print("empiezo a entrenar titulo")
titulo_pca = entrenar_pca_para_texto(df_train["title"], modelo_embeddings, componentes_conservar = 20)
print("empiezo a guardar")
joblib.dump(titulo_pca, "titulo_pca.joblib")
print("termino de guardar")

#Set de entrenamiento:
titulo_emb_train = generar_embeddings_pca(df_train["title"], modelo_embeddings, titulo_pca)
print("termino embedding train")
#Set de validacion:
titulo_emb_val = generar_embeddings_pca(df_val["title"], modelo_embeddings, titulo_pca)
print("termino embedding val")
#Set de testeo:
titulo_emb_test = generar_embeddings_pca(df_test["title"], modelo_embeddings, titulo_pca)
print("termino embedding test")

print("guardo titulos")
np.save("titulo_emb_train.npy", titulo_emb_train)
np.save("titulo_emb_val.npy", titulo_emb_val)
np.save("titulo_emb_test.npy", titulo_emb_test)
print("TERMINEEEEEEEEEEEEEEEE")