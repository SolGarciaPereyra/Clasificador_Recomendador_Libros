import pandas as pd
import torch
import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer

#LEO DATASETS
chunks_train = pd.read_json("set_train_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_train:
    chunk_list.append(chunk)
df_train = pd.concat(chunk_list, ignore_index=True)

chunks_val = pd.read_json("set_val_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_val:
    chunk_list.append(chunk)
df_val = pd.concat(chunk_list, ignore_index=True)

chunks_test = pd.read_json("set_test_1.json", lines=True, chunksize=100_000)
chunk_list = []
for chunk in chunks_test:
    chunk_list.append(chunk)
df_test = pd.concat(chunk_list, ignore_index=True)


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


device = "cuda" if torch.cuda.is_available() else "cpu"

#pca descripcion
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2", device=device)
descripcion_pca = entrenar_pca_para_texto(df_train["description"], modelo_embeddings, componentes_conservar = 50)
joblib.dump(descripcion_pca, "descripcion_pca.joblib") #guardo el pca

#embeddings descripcion
descripcion_emb_train = generar_embeddings_pca(df_train["description"], modelo_embeddings, descripcion_pca)
descripcion_emb_val = generar_embeddings_pca(df_val["description"], modelo_embeddings, descripcion_pca)
descripcion_emb_test = generar_embeddings_pca(df_test["description"], modelo_embeddings, descripcion_pca)

#guardo los embeddings
np.save("descripcion_emb_train.npy", descripcion_emb_train)
np.save("descripcion_emb_val.npy", descripcion_emb_val)
np.save("descripcion_emb_test.npy", descripcion_emb_test)
