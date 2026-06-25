from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluar_metricas_regresion(y_pred, y):
    MSE = mean_squared_error(y, y_pred)
    RMSE = root_mean_squared_error(y, y_pred)
    MAE = mean_absolute_error(y, y_pred)
    R2 = r2_score(y, y_pred)
    return MSE, RMSE, MAE, R2

def evaluar_metricas_clasificacion(y_pred, y):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average = "macro", zero_division = 0)
    recall = recall_score(y, y_pred, average = "macro", zero_division = 0)
    f1 = f1_score(y, y_pred, average = "macro", zero_division = 0)
    return accuracy, precision, recall, f1

def tabla_metricas(metricas_train, metricas_val, nombres_metricas):
    return pd.DataFrame({"Train": metricas_train, "Validation": metricas_val}, index=nombres_metricas)
