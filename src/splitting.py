from sklearn.model_selection import StratifiedShuffleSplit

def split_estratificado_autor(df, test_size=0.2, val_size=0.2):
    """
    - garantiza que ningun autor aparezca en mas de un split
    - intenta mantener la distribución de "estrellas" balanceada entre splits,
      estratificando segun su clase dominante de estrellas de cada autor
    """
    #Por cada autor, calcula su clase de estrellas dominante
    autor_summary = df.groupby("author_id_principal")["estrellas"].apply(lambda x: x.mode().iloc[0]).reset_index(name = "clase_dominante")
    #en este df, cada fila es un autor (una fila por cada autor diferente)

    #StratifiedShuffleSplit falla si una categoria de estrellas tiene menos
    #de dos autores asignados a ella. Se necesita al menos un autor representante
    #en cada split. Los agrupamos en una categria q llamamos "rara"
    clase_counts = autor_summary["clase_dominante"].value_counts()
    clases_raras = clase_counts[clase_counts < 2].index
    mapeo = {c: "rara" for c in clases_raras}
    autor_summary["clase_split"] = autor_summary["clase_dominante"].replace(mapeo)

    #Convierte la feature "author_id_principal" y "clase_split" del df de autores en un array de numpy
    autores = autor_summary["author_id_principal"].values 
    clase_splits = autor_summary["clase_split"].values 

    #Primer split: separamos test en funcion de los autores, estratificado por la clase dominante (+ rara)
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    #autores es la lista de cosas a dividir, y clase_splits la etiqueta de clase que sirve para el split estratificado
    for dev_idx, test_idx in split_1.split(autores, clase_splits):
        break  #agarramos el primer (y unico) split, y salimos del loop    

    autores_dev = autores[dev_idx]
    clase_splits_dev = clase_splits[dev_idx]

    #Segundo split: separamos val de train
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_idx, val_idx in split_2.split(autores_dev, clase_splits_dev):
        break   

    autores_train = set(autores_dev[train_idx])
    autores_val = set(autores_dev[val_idx])
    autores_test = set(autores[test_idx])

    #Mapeamos los autores a las filas del df original
    df_train = df[df["author_id_principal"].isin(autores_train)].copy()
    df_val = df[df["author_id_principal"].isin(autores_val)].copy()
    df_test = df[df["author_id_principal"].isin(autores_test)].copy()

    return df_train, df_val, df_test