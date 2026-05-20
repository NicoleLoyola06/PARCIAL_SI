import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar dataset
df = pd.read_csv("data/sismos_igp.csv")

# 2. Normalizar nombres de columnas
df.columns = df.columns.str.lower().str.strip()

# 3. Verificar columnas necesarias
columnas_necesarias = ["latitud", "longitud", "profundidad", "magnitud"]

for columna in columnas_necesarias:
    if columna not in df.columns:
        raise ValueError(f"Falta la columna obligatoria: {columna}")

# 4. Eliminar valores nulos
df = df.dropna(subset=columnas_necesarias)

# 5. Seleccionar variables predictoras y variable objetivo
X = df[["latitud", "longitud", "profundidad"]]
y = df["magnitud"]

# 6. Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Crear modelo inteligente
modelo = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# 8. Entrenar modelo
modelo.fit(X_train, y_train)

# 9. Realizar predicciones
y_pred = modelo.predict(X_test)

# 10. Evaluar modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo")
print("----------------------")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# 11. Guardar modelo entrenado
joblib.dump(modelo, "modelo/modelo_sismos.pkl")

print("Modelo guardado correctamente en modelo/modelo_sismos.pkl")
