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

def evaluate_metrics_pytorch_clasificacion(y_pred_train, y_train, y_pred_val, y_val):
    accuracy_train, precision_train, recall_train, f1_train = evaluar_metricas_clasificacion(y_pred_train, y_train)
    accuracy_val, precision_val, recall_val, f1_val = evaluar_metricas_clasificacion(y_pred_val, y_val)

    print(f"Accuracy train: {accuracy_train:.4f}")
    print(f"Accuracy val: {accuracy_val:.4f}")

    print(f"Precision train; {precision_train:.4f}")
    print(f"Precision val: {precision_val:.4f}")

    print(f"Recall train; {recall_train:.4f}")
    print(f"Recall val: {recall_val:.4f}")

    print(f"F1 train; {f1_train:.4f}")
    print(f"F1 val: {f1_val:.4f}")

    dict_train = {"accuracy": accuracy_train, "recall_macro": recall_train, "precision_macro": precision_train, "f1_macro": f1_train}
    dict_val = {"accuracy": accuracy_val, "recall_macro": recall_val, "precision_val": precision_val, "f1_macro": f1_val}
    return dict_train, dict_val

def evaluate_metrics_pytorch_regresion(y_pred_train, y_train, y_pred_val, y_val):
    mse_train, rmse_train, mae_train, r2_train = evaluar_metricas_regresion(y_pred_train, y_train)
    mse_val, rmse_val, mae_val, r2_val = evaluar_metricas_regresion(y_pred_val, y_val)

    print(f"MSE train: {mse_train:.4f}")
    print(f"MSE val: {mse_val:.4f}")

    print(f"RMSE train: {rmse_train:.4f}")
    print(f"RMSE val: {rmse_val:.4f}")

    print(f"MAE train: {mae_train:.4f}")
    print(f"MAE val: {mae_val:.4f}")

    print(f"R2 train: {r2_train:.4f}")
    print(f"R2 val: {r2_val:.4f}")

    dict_train = {"MSE": mse_train, "RMSE": rmse_train, "MAE": mae_train, "R2": r2_train}
    dict_val = {"MSE": mse_val, "RMSE": rmse_val, "MAE": mae_val, "R2": r2_val}
    return dict_train, dict_val