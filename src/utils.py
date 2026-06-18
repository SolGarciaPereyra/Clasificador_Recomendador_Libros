from src.features import limpiar_features_no_necesarias

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
