import pandas as pd
import numpy as np
from src.modelos import MLP_regresion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, f1_score, r2_score, precision_score
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor, XGBClassifier
from IPython.display import display

from src.features import limpiar_features_no_necesarias
from src.metricas import evaluar_metricas_clasificacion, evaluar_metricas_regresion, evaluar_metricas_clasificacion_binaria
from src.graficos import plot_losses, plot_confusion_matrix
from src.metricas import tabla_metricas

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
    """Fusiona libros con autor y titulo iguales, gestiona la fusion de las otras features del libro"""
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

def mini_grid_search_logistica(X_train_scaled, y_train_clasificacion, X_val_scaled, y_val_clasificacion, pesos_clase_1, pesos_clase_2, peso_clase_3, peso_clase_4, pesos_clase_5):
    resultados = []

    for peso_1, peso_2, peso_5 in product(pesos_clase_1, pesos_clase_2, pesos_clase_5):
        pesos = {1: peso_1, 2: peso_2, 3: peso_clase_3, 4: peso_clase_4, 5: peso_5}
        modelo = LogisticRegression(max_iter = 1000, class_weight = pesos)
        modelo.fit(X_train_scaled, y_train_clasificacion)
        y_pred_val = modelo.predict(X_val_scaled)
        accuracy, precision, recall, f1 = evaluar_metricas_clasificacion(y_pred_val, y_val_clasificacion)
        resultados.append({"peso_1": peso_1, "peso_2": peso_2, "peso_5": peso_5, "accuracy": accuracy, 
                           "precision": precision, "recall": recall, "f1": f1})
        
    return resultados

def mini_grid_search_logistica_undersampling(X_train, y_train_clasificacion, X_val, y_val_clasificacion, cantidades_clase_4, pesos_clase_1, pesos_clase_2, peso_clase_3, peso_clase_4, pesos_clase_5):
    resultados = []

    for cantidad in cantidades_clase_4:
        undersampling = RandomUnderSampler(sampling_strategy={4: cantidad}, random_state=42)
        X_res, y_res = undersampling.fit_resample(X_train, y_train_clasificacion)
        scaler = StandardScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        X_val_scaled = scaler.transform(X_val)

        for peso_1, peso_2, peso_5 in product(pesos_clase_1, pesos_clase_2, pesos_clase_5):
            pesos = {1: peso_1, 2: peso_2, 3: peso_clase_3, 4: peso_clase_4, 5: peso_5}
            modelo = LogisticRegression(max_iter = 1000, class_weight=pesos)
            modelo.fit(X_res_scaled, y_res)
            pred = modelo.predict(X_val_scaled)
            accuracy, precision, recall, f1 = evaluar_metricas_clasificacion(pred, y_val_clasificacion)
            resultados.append({"cantidad": cantidad, "peso_1": peso_1, "peso_2": peso_2, "peso_5": peso_5,
                "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

    return resultados

def make_dataloaders(X_train, y_train, X_val, y_val, n_batches=64, y_dtype = torch.long):
    """Transforms numpy array into tensor"""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=y_dtype)
    X_vl = torch.tensor(X_val, dtype=torch.float32)
    y_vl = torch.tensor(y_val, dtype=y_dtype)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_vl, y_vl)

    batch_size = max(1, len(X_train) // n_batches)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 2, persistent_workers = True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory = True, num_workers = 2, persistent_workers = True)

    return train_loader, val_loader

def run_experiment_pytorch_clasificacion(model, model_name, X_train, y_train, X_val, y_val, device, learning_rate = 0.001, epochs = 100, pesos = None, save = False):
    """Nuestro target de clasificacion multiclase empieza en 1, pero para entrenar se necesita que empiece en 0,
    hago un par de modificaciones para soportar esto. """
    model = model.to(device)
    #temporalmente cambio las clases de 1-5 a 0-4 por la loss
    y_train_temp = y_train.astype(np.int64) - 1
    y_val_temp = y_val.astype(np.int64) - 1

    #rebalanceo:
    if pesos is None:
        counts = np.bincount(y_train_temp)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
    else:
        #transformo de {1: , 2: , 3: , 4: , 5: } a lista
        weights = [pesos[1], pesos[2], pesos[3], pesos[4], pesos[5]]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    #entreno con los valores de clase temporales
    train_loader, val_loader = make_dataloaders(X_train, y_train_temp, X_val, y_val_temp, n_batches = 64)
    train_losses, val_losses = model.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, waiting = 5)

    #le sumo 1 a la prediccion para volver a las clases originales
    y_pred_train = model.predict(X_train, device) + 1
    y_pred_val = model.predict(X_val, device) + 1
    train_metrics = evaluar_metricas_clasificacion(y_pred_train, y_train)
    val_metrics = evaluar_metricas_clasificacion(y_pred_val, y_val)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_val = confusion_matrix(y_val, y_pred_val)
    plot_confusion_matrix(cm_train, cm_val)
    plot_losses(train_losses, val_losses, f"{model_name} - pytorch")

    if save is True:
        torch.save(model.state_dict(), "modelos/mlp_clasificacion.pth")

    return train_metrics, val_metrics

def run_grid_search_pytorch_clasificacion(modelo, learning_rate, model_name, X_train, y_train, X_val, y_val, device, epochs = 100, pesos = None):
    modelo = modelo.to(device)
    y_train_temp = y_train.astype(np.int64) - 1
    y_val_temp = y_val.astype(np.int64) - 1

    #rebalanceo:
    if pesos is None:
        counts = np.bincount(y_train_temp)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
    else:
        #transformo de {1: , 2: , 3: , 4: , 5: } a lista
        weights = [pesos[1], pesos[2], pesos[3], pesos[4], pesos[5]]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    
    #entreno con las clases temporales de 0 a 4
    train_loader, val_loader = make_dataloaders(X_train, y_train_temp, X_val, y_val_temp, n_batches=64)
    modelo.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs)
    
    #sumo 1 a las predicciones para volver a las clases de 1 a 5
    y_pred_val = modelo.predict(X_val, device) + 1
    f1_val = f1_score(y_val, y_pred_val, average="macro")
    return f1_val

def run_experiment_pytorch_regresion(model, model_name, X_train, y_train, X_val, y_val, device, learning_rate = 0.001, epochs = 100):
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, n_batches = 64, y_dtype = torch.float32)
    train_losses, val_losses = model.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs = epochs, waiting = 5)

    y_pred_train = model.predict(X_train, device)
    y_pred_val = model.predict(X_val, device)
    train_metrics = evaluar_metricas_regresion(y_pred_train, y_train)
    val_metrics = evaluar_metricas_regresion(y_pred_val, y_val)

    plot_losses(train_losses, val_losses, f"{model_name} - pytorch")
    return train_metrics, val_metrics

def run_grid_search_pytorch_regresion(modelo, learning_rate, model_name, X_train, y_train, X_val, y_val, device, epochs = 100):
    modelo = modelo.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    
    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, n_batches=64, y_dtype=torch.float32)
    modelo.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs)
    
    y_pred_val = modelo.predict(X_val, device)
    r2_val = r2_score(y_val, y_pred_val)
    return r2_val

def permutation_importance_pytorch(model, X, y, feature_names, device, metric_fn = mean_squared_error, higher_is_better=False, n_repeats=5, random_state=42, clasificacion = False):
    """Pytorch no tiene una funcion integrada para saber la feature importance del modelo, por lo que
    se permuta cada feature (de a una) para ver como afecta en la prediccion. """
    generador_numeros_aleatorios = np.random.default_rng(random_state)

    #defino el baseline para ver si al permutar la columna (para descorrelacionar sus valores con la prediccion) afecta
    if clasificacion is True: #modifique las clases en el entrenamiento asi que tengo que sumar 1 a la prediccion
        y_pred_base = model.predict(X, device) + 1
    else: #para regresion no modifico las predicciones
        y_pred_base = model.predict(X, device)

    baseline_score = metric_fn(y, y_pred_base)

    means = []
    for col in range(X.shape[1]):
        original_col = X[:, col].copy() #copio la columna original
        scores = []
        for _ in range(n_repeats): #repito para estabilidad
            generador_numeros_aleatorios.shuffle(X[:, col])  #mezlco la columna
            y_pred_perm = model.predict(X, device)
            scores.append(metric_fn(y, y_pred_perm))
        X[:, col] = original_col  #restauro la columna a su estado original
        means.append(np.mean(scores))

    means = np.array(means)
    if higher_is_better: #para clasificacion queremos maximizar las metricas pero para regresion queremos minimizar
        importances = baseline_score - means
    else:
        importances = means - baseline_score

    df_importance = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_importance

def grid_search_rf_clasificacion(pesos, X, y, parametros_grid_search_rf):
    RandomForestGridSearchClasificacion = RandomForestClassifier(random_state = 42, class_weight = pesos)

    grid = GridSearchCV(estimator = RandomForestGridSearchClasificacion, param_grid = parametros_grid_search_rf, cv = 3, scoring = "f1_macro", n_jobs = -1, verbose = 2)
    grid.fit(X, y)

    mejores_parametros = grid.best_params_
    print(f"Mejores parametros: {mejores_parametros}")
    print(f"Mejor F1 CV: {grid.best_score_}")
    return mejores_parametros

def evaluar_modelo_rf_clasificacion(mejores_parametros, pesos, X_train, y_train, X_val, y_val):
    modeloRandomForestClasificacion = RandomForestClassifier(**mejores_parametros, random_state = 42, class_weight = pesos)
    modeloRandomForestClasificacion.fit(X_train, y_train)

    y_pred_train_modeloRandomForestClasificacion = (modeloRandomForestClasificacion.predict(X_train))
    y_pred_val_modeloRandomForestClasificacion = (modeloRandomForestClasificacion.predict(X_val))

    metricas_train = evaluar_metricas_clasificacion(y_pred_train_modeloRandomForestClasificacion, y_train)
    metricas_val = evaluar_metricas_clasificacion(y_pred_val_modeloRandomForestClasificacion, y_val)
    tabla_metricas_rf_clasificacion = tabla_metricas(metricas_train, metricas_val, ["Accuracy", "Precision", "Recall", "F1"])
    display(tabla_metricas_rf_clasificacion)

    cm_train = confusion_matrix(y_train, y_pred_train_modeloRandomForestClasificacion)
    cm_val = confusion_matrix(y_val, y_pred_val_modeloRandomForestClasificacion)
    plot_confusion_matrix(cm_train, cm_val)
    return tabla_metricas_rf_clasificacion, modeloRandomForestClasificacion

def grid_search_xgb_clasificacion(pesos, X, y, parametros_grid_search_xgb):
    pesos_transformados_sub = np.array([pesos[y_i] for y_i in y])
    XGBoostGridSearchClasificacion = XGBClassifier(objective = "multi:softmax", num_class = 5, device = "cuda", tree_method = "hist", random_state = 42) 

    #uso los mismos indices para obtener los subconjuntos
    grid = GridSearchCV(estimator = XGBoostGridSearchClasificacion, param_grid = parametros_grid_search_xgb, cv = 3, scoring = "f1_macro", n_jobs = 1, verbose = 2)
    grid.fit(X, y, sample_weight = pesos_transformados_sub)

    mejores_parametros = grid.best_params_
    print(f"Mejores parametros: {mejores_parametros}")
    print(f"Mejor F1 CV: {grid.best_score_}")
    return mejores_parametros

def evaluar_modelo_xgb_clasificacion(mejores_parametros, pesos, X_train, y_train, X_val, y_val):
    pesos_transformados_train = np.array([pesos[y] for y in y_train])
    modeloXGBoostClasificacion = XGBClassifier(**mejores_parametros, objective = "multi:softmax", num_class = 5, device = "cuda", tree_method = "hist", random_state = 42) 
    modeloXGBoostClasificacion.fit(X_train, y_train, sample_weight = pesos_transformados_train) #XGBoost no requiere obligatoriamnete que se escalen los valores, pero lo hag para poder comprar la feature importance

    y_pred_train = (modeloXGBoostClasificacion.predict(X_train))
    y_pred_val = (modeloXGBoostClasificacion.predict(X_val))
    #volvemos a las clases originales sumando 1
    y_train = y_train + 1
    y_pred_train_modeloXGBoostClasificacion = y_pred_train + 1
    y_pred_val_modeloXGBoostClasificacion = y_pred_val + 1

    #calculo las metricas usando las clases originales entre 1 y 5
    metricas_train = evaluar_metricas_clasificacion(y_pred_train_modeloXGBoostClasificacion, y_train)
    metricas_val = evaluar_metricas_clasificacion(y_pred_val_modeloXGBoostClasificacion, y_val)
    tabla_metricas_xgboost_clasificacion = tabla_metricas(metricas_train, metricas_val, ["Accuracy", "Precision", "Recall", "F1"])
    print(tabla_metricas_xgboost_clasificacion)

    cm_train = confusion_matrix(y_train, y_pred_train_modeloXGBoostClasificacion)
    cm_val = confusion_matrix(y_val, y_pred_val_modeloXGBoostClasificacion)
    plot_confusion_matrix(cm_train, cm_val)
    return tabla_metricas_xgboost_clasificacion, modeloXGBoostClasificacion

def grid_search_xgb_aceptacion(pesos, X, y, parametros_grid_search_xgb):
    pesos_transformados_sub = np.array([pesos[y_i] for y_i in y])
    XGBoostGridSearchClasificacion = XGBClassifier(objective = "binary:logistic", device = "cuda", tree_method = "hist", random_state = 42) 

    #uso los mismos indices para obtener los subconjuntos
    grid = GridSearchCV(estimator = XGBoostGridSearchClasificacion, param_grid = parametros_grid_search_xgb, cv = 3, scoring = "precision", n_jobs = 1, verbose = 1, error_score = "raise")
    grid.fit(X, y, sample_weight = pesos_transformados_sub)

    mejores_parametros = grid.best_params_
    print(f"Mejores parametros: {mejores_parametros}")
    print(f"Mejor precision CV: {grid.best_score_}")
    return mejores_parametros

def evaluar_modelo_xgb_aceptacion(mejores_parametros, pesos, X_train, y_train, X_val, y_val):
    pesos_transformados_train = np.array([pesos[y] for y in y_train])
    modeloXGBoostClasificacion = XGBClassifier(**mejores_parametros, objective = "binary:logistic", device = "cuda", tree_method = "hist", random_state = 42) 
    modeloXGBoostClasificacion.fit(X_train, y_train, sample_weight = pesos_transformados_train) #XGBoost no requiere obligatoriamnete que se escalen los valores, pero lo hag para poder comprar la feature importance

    y_pred_train = (modeloXGBoostClasificacion.predict(X_train))
    y_pred_val = (modeloXGBoostClasificacion.predict(X_val))

    #calculo las metricas usando las clases originales entre 1 y 5
    metricas_train = evaluar_metricas_clasificacion_binaria(y_pred_train, y_train)
    metricas_val = evaluar_metricas_clasificacion_binaria(y_pred_val, y_val)
    tabla_metricas_xgboost_clasificacion = tabla_metricas(metricas_train, metricas_val, ["Accuracy", "Precision", "Recall", "F1"])
    print(tabla_metricas_xgboost_clasificacion)

    cm_train = confusion_matrix(y_train, y_pred_train, labels = [0, 1])
    cm_val = confusion_matrix(y_val, y_pred_val, labels = [0, 1])
    plot_confusion_matrix(cm_train, cm_val, labels = [0, 1])
    return tabla_metricas_xgboost_clasificacion, modeloXGBoostClasificacion, metricas_train, metricas_val

def run_grid_search_pytorch_aceptacion(modelo, learning_rate, model_name, X_train, y_train, X_val, y_val, device, epochs = 100, pesos = None):
    #rebalanceo: transformo {0: ..., 1: ...} a lista
    weights = [pesos[0], pesos[1]]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    train_loader, val_loader = make_dataloaders(X_train, y_train.astype(np.int64), X_val, y_val.astype(np.int64), n_batches=64)

    modelo.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs)
    y_pred_val = modelo.predict(X_val, device)
    precision_val = precision_score(y_val, y_pred_val, average="binary", zero_division=0)

    return precision_val

def run_experiment_pytorch_aceptacion(model, model_name, X_train, y_train, X_val, y_val, device, learning_rate = 0.001, epochs = 100, pesos = None):
    #rebalanceo: transformo {0: ..., 1: ...} a lista
    weights = [pesos[0], pesos[1]]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, val_loader = make_dataloaders(X_train, y_train.astype(np.int64), X_val, y_val.astype(np.int64), n_batches=64)

    train_losses, val_losses = model.train_model(train_loader, val_loader, optimizer, loss_fn, device, epochs=epochs, waiting=5)
    y_pred_train = model.predict(X_train, device)
    y_pred_val = model.predict(X_val, device)

    train_metrics = evaluar_metricas_clasificacion_binaria(y_pred_train, y_train)
    val_metrics = evaluar_metricas_clasificacion_binaria(y_pred_val, y_val)

    cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
    cm_val = confusion_matrix(y_val, y_pred_val, labels=[0, 1])
    plot_confusion_matrix(cm_train, cm_val, labels = [0, 1])

    return train_metrics, val_metrics