import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier


def plot_rating(df): #
    plt.hist(df["average_rating"], bins=50, edgecolor="white")
    plt.xlabel("Rating Promedio")
    plt.ylabel("Cantidad de libros")
    plt.title("Distribución de ratings", fontweight = "bold")
    plt.show()

def plot_boxplot_rating(df):
    ratings = df["average_rating"]
    plt.boxplot(ratings,vert=True,patch_artist=True)

    plt.ylabel("Rating promedio")
    plt.title("Distribución de ratings", fontweight = "bold")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stars(df): #
    porcentajes = (df["estrellas"].value_counts(normalize = True).sort_index()*100)
    bars = plt.bar(porcentajes.index, porcentajes.values)
    
    plt.xlabel("Estrellas")
    plt.ylabel("% de libros")
    plt.title("Distribución de estrellas", fontweight = "bold")
    
    for bar in bars:
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, altura,f"{altura:.1f}%", ha="center", va="bottom")
    plt.show()

def plot_generos_mas_frecuentes(contador_generos): #
    contador_ordenado = contador_generos.sort_values()
    plt.figure(figsize=(10, 6))
    plt.barh(contador_ordenado.index, contador_ordenado.values)

    plt.title("Géneros más frecuentes", fontweight = "bold")
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
    plt.title("Rating promedio por género", fontweight = "bold")
    plt.tight_layout()
    plt.show()

def plot_boxplot_rating_por_genero(df): #
    plt.figure(figsize=(12, 6))
    df_generos = df.explode("genres")
    df_generos.boxplot(column="average_rating",by="genres",rot=90)

    plt.xlabel("Género")
    plt.ylabel("Average Rating")
    plt.title("Distribución de ratings por género", fontweight = "bold")
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
    plt.title("Distribución de géneros por libro", fontweight = "bold")

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

def plot_boxplot_rating_por_autor_frecuent(df_top, titulo):
    plt.figure(figsize = (10, 6))
    sns.boxplot(data = df_top, x = "author_id_principal", y = "average_rating")
    plt.ylim(0, 5)
    plt.xlabel("Autor")
    plt.ylabel("Average rating")
    plt.title(titulo, fontweight = "bold")
    plt.xticks(rotation = 45)
    plt.show()

def plot_correlacion_features(corr_rating):
    plt.figure(figsize=(8, max(8, len(corr_rating) * 0.3)))
    sns.heatmap(corr_rating.to_frame(),annot=True,cmap="coolwarm",center=0,fmt=".2f")
    plt.title("Correlación de las features con average_rating", fontweight = "bold")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm_train, cm_val, title = "Matriz de confusion"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_train, annot=True, fmt="d", cmap = "Blues", ax=axes[0])
    axes[0].set_title("Train", fontweight = "bold")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")

    sns.heatmap(cm_val, annot=True, fmt="d", cmap = "Oranges", ax=axes[1])
    axes[1].set_title("Validation", fontweight = "bold")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_boxplot_rating_por_cant_generos(df_us_en_1_filtro):
    #Grafico la relacion entre cantidad_generos y average_rating:
    plt.figure(figsize = (10, 6))
    sns.boxplot(data = df_us_en_1_filtro, x = "cantidad_generos", y = "average_rating")
    plt.xlabel("Cantidad de géneros")
    plt.ylabel("Average rating")
    plt.title("Relación entre cantidad de géneros y rating", fontweight = "bold")
    plt.show()

def plot_learning_curves_linear_regression(X_train_scaled, y_train_regresion, X_val_scaled, y_val_regresion):
    #Reordenamos los indices de X_train (de manera random), de esta manera, cada subconjunto de datos va a contener al anterior
    np.random.seed(42)
    indices = np.random.permutation(len(X_train_scaled)) 
    #Elegimos el porcentaje de datos sobre los que vamos a evaluar la regresion
    porcentajes = np.linspace(0.1, 1.0, 10)
    rmse_train = []
    rmse_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train_scaled) * porcentaje)
        X_sub = X_train_scaled[indices[:cant_muestras]]
        y_sub = y_train_regresion[indices[:cant_muestras]]
        #Creamos y entrenamos el modelo de Regresion Lineal:
        modelo = LinearRegression()
        modelo.fit(X_sub, y_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo.predict(X_val_scaled)
        #Calculamos ambos errores:
        rmse_train.append(np.sqrt(mean_squared_error(y_sub, pred_train)))
        rmse_val.append(np.sqrt(mean_squared_error(y_val_regresion, pred_val)))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6, 6))
    plt.plot(porcentajes, rmse_train, label = "Train")
    plt.plot(porcentajes, rmse_val, label = "Validacion")
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("RMSE")
    plt.title("Learning Curve - Regresión Lineal", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()

def plot_learning_curves_logistic_regression(X_train_scaled, y_train_clasificacion, X_val_scaled, y_val_clasificacion, tecnica_rebalanceo, modelo_class_weight = None):
    #Reordenamos los indices de X_train (de manera random), de esta manera, cada subconjunto de datos va a contener al anterior
    np.random.seed(42)
    indices = np.random.permutation(len(X_train_scaled)) 
    #Elegimos el porcentaje de datos sobre los que vamos a evaluar la regresion
    porcentajes = np.linspace(0.1, 1.0, 10)
    f1_train = []
    f1_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train_scaled) * porcentaje)
        X_sub = X_train_scaled[indices[:cant_muestras]]
        y_sub = y_train_clasificacion[indices[:cant_muestras]]
        #Creamos y entrenamos el modelo de Regresion Logistica:
        modelo_actual = LogisticRegression(max_iter = 1000, class_weight = modelo_class_weight)
        modelo_actual.fit(X_sub, y_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo_actual.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo_actual.predict(X_val_scaled)
        #Calculamos ambos errores:
        f1_train.append(f1_score(y_sub, pred_train, average = "macro"))
        f1_val.append(f1_score(y_val_clasificacion, pred_val, average = "macro"))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6, 6))
    plt.plot(porcentajes, f1_train, label = 'Train')
    plt.plot(porcentajes, f1_val, label = 'Validation')
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("F1 Score Macro")
    plt.title(f"Learning Curve - Regresión Logística + {tecnica_rebalanceo}", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()

def plot_learning_curves_xgboost_regression(X_train, y_train, X_val, y_val, n_estimators = 300, max_depth = 4, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, objective = "reg:squarederror", random_state = 42):
    porcentajes = np.linspace(0.1, 1.0, 10)
    rmse_train = []
    rmse_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train) * porcentaje)
        indices = np.random.choice(len(X_train), size = cant_muestras, replace = False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]
        #Creamos y entrenamos el modelo de XGBoost para regresion:
        modelo = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree, objective = objective, random_state = random_state)
        modelo.fit(X_sub, y_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo.predict(X_val)
        #Calculamos ambos errores:
        rmse_train.append(np.sqrt(mean_squared_error(y_sub, pred_train)))
        rmse_val.append(np.sqrt(mean_squared_error(y_val, pred_val)))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6,6))
    plt.plot(porcentajes, rmse_train, label = "Train")
    plt.plot(porcentajes, rmse_val,label = "Validation")
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("RMSE")
    plt.title("Learning Curve - XGBoost para Regresion", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()
    
def plot_learning_curves_xgboost_classification(X_train, y_train, X_val, y_val, pesos = None, n_estimators = 300, max_depth = 4, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, objective = "multi:softmax", random_state = 42):
    porcentajes = np.linspace(0.1, 1.0, 10)
    f1_train = []
    f1_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train) * porcentaje)
        indices = np.random.choice(len(X_train), size = cant_muestras, replace = False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]
        #Creamos y entrenamos el modelo de XGBoost para regresion:
        modelo = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, colsample_bytree = colsample_bytree, objective = objective, num_class = len(np.unique(y_train)), random_state = random_state)
        pesos_sub = pesos[indices]
        modelo.fit(X_sub, y_sub, sample_weight = pesos_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo.predict(X_val)
        #Calculamos ambos errores:
        f1_train.append(f1_score(y_sub, pred_train, average = "macro"))
        f1_val.append(f1_score(y_val, pred_val, average = "macro"))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6, 6))
    plt.plot(porcentajes, f1_train, label = "Train")
    plt.plot(porcentajes, f1_val,label = "Validation")
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("F1 Score Macro")
    plt.title("Learning Curve - XGBoost para Clasificacion", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()


def plot_learning_curves_randomforest_regression(X_train, y_train, X_val, y_val, n_estimators = 300, max_depth = 4, min_samples_split = 2, min_samples_leaf = 1, max_features = "sqrt", random_state = 42):
    porcentajes = np.linspace(0.1, 1.0, 10)
    rmse_train = []
    rmse_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train) * porcentaje)
        indices = np.random.choice(len(X_train), size = cant_muestras, replace = False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]
        #Creamos y entrenamos el modelo de Random Forest para regresion:
        modelo = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state)        
        modelo.fit(X_sub, y_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo.predict(X_val)
        #Calculamos ambos errores:
        rmse_train.append(np.sqrt(mean_squared_error(y_sub, pred_train)))
        rmse_val.append(np.sqrt(mean_squared_error(y_val, pred_val)))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6,6))
    plt.plot(porcentajes, rmse_train, label = "Train")
    plt.plot(porcentajes, rmse_val,label = "Validation")
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("RMSE")
    plt.title("Learning Curve - Random Forest para Regresion", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()


def plot_learning_curves_randomforest_classification(X_train, y_train, X_val, y_val, pesos = None, n_estimators = 300, max_depth = 4, min_samples_split = 2, min_samples_leaf = 1, max_features = "sqrt", random_state = 42):
    porcentajes = np.linspace(0.1, 1.0, 10)
    f1_train = []
    f1_val = []
    for porcentaje in porcentajes:
        #Calculamos cuantas muestras corresponden a este porcentaje:
        cant_muestras = int(len(X_train) * porcentaje)
        indices = np.random.choice(len(X_train), size = cant_muestras, replace = False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]
        #Creamos y entrenamos el modelo de Random Forest para regresion:
        modelo = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state, class_weight = pesos)       
        modelo.fit(X_sub, y_sub)
        #Predecimos sobre el mismo conjunto de entrenamiento (para ver el error de train):
        pred_train = modelo.predict(X_sub)
        #Y predecimos sobre el conjunto de validacion (para ver el error de val):
        pred_val = modelo.predict(X_val)
        #Calculamos ambos errores:
        f1_train.append(f1_score(y_sub, pred_train, average = "macro"))
        f1_val.append(f1_score(y_val, pred_val, average = "macro"))

    #Ahora graficamos las curvas:
    plt.figure(figsize = (6,6))
    plt.plot(porcentajes, f1_train, label = "Train")
    plt.plot(porcentajes, f1_val,label = "Validation")
    plt.xlabel("Porcentaje del dataset de entrenamiento")
    plt.ylabel("F1 Score Macro")
    plt.title("Learning Curve - Random Forest para Clasificacion", fontweight = "bold")
    plt.legend()
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.show()

def plot_comparacion_metricas(metricas_dict, title):
    """metricas_dict = {"Sin rebalanceo": (acc, prec, rec, f1), "Balanced": (acc, prec, rec, f1)...}"""

    nombres_metricas = ["Accuracy", "Precision", "Recall", "F1"]
    df = pd.DataFrame(metricas_dict, index = nombres_metricas)
    df.T.plot(kind="bar", figsize=(10, 5))

    plt.title(title, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.show()

def plot_feature_importance_regression(coeficientes, modelo_regresor):
    top = coeficientes.head(20)
    plt.figure(figsize = (6, 6))
    plt.barh(top["feature"], top["abs_coef"])
    plt.xlabel("|Coeficiente|")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - Regresion {modelo_regresor}", fontweight = "bold")
    plt.gca().invert_yaxis()
    plt.show()

def plot_feature_importance_xgb(importancias, modelo_xgb):
    top = importancias.head(20)
    plt.figure(figsize = (6, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - XGBoost {modelo_xgb}", fontweight = "bold")
    plt.gca().invert_yaxis()
    plt.show()

def plot_feature_importance_rf(importancias, modelo_rf):
    top = importancias.head(20)
    plt.figure(figsize = (6, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - Random Forest {modelo_rf}", fontweight = "bold")
    plt.gca().invert_yaxis()
    plt.show()

def plot_feature_importance_pytorch(df_importance, top_n=20, title="Feature Importance"):
    df_plot = df_importance.head(top_n).iloc[::-1]  # invierto para que la más importante quede arriba
    plt.figure(figsize=(8, max(4, 0.3 * len(df_plot))))
    plt.barh(df_plot["feature"], df_plot["importance"])
    plt.xlabel("Aumento del error al permutar" )
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses, name_model):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Cross-Entropy")
    plt.plot(val_losses, label="Val Cross-Entropy")
    plt.xlabel("Época")
    plt.ylabel("Cross-Entropy")
    plt.title(f"Evolución de la función de costo — Modelo {name_model}")
    plt.legend()
    plt.tight_layout()
    plt.show()