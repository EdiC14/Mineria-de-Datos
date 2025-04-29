import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar datos
df = pd.read_csv('googleplaystore_limpio.csv')

# Asegurarse de que Last Updated sea tipo fecha
df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors='coerce')

# Eliminar valores nulos
df = df.dropna(subset=["Last Updated"])

# Crear columna Año-Mes
df["YearMonth"] = df["Last Updated"].dt.to_period('M')

# Contar número de Apps actualizadas cada mes
apps_por_mes = df.groupby("YearMonth").size().reset_index(name="Apps_Actualizadas")

# Convertir YearMonth a números para regresión
apps_por_mes["YearMonth_num"] = np.arange(len(apps_por_mes))

# Separar variables
X = apps_por_mes[["YearMonth_num"]]
y = apps_por_mes["Apps_Actualizadas"]

# Crear modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Predecir valores
y_pred = modelo.predict(X)

# Graficar
plt.figure(figsize=(10,6))
plt.plot(apps_por_mes["YearMonth"].astype(str), y, label="Apps Reales", marker='o')
plt.plot(apps_por_mes["YearMonth"].astype(str), y_pred, label="Predicción Lineal", linestyle='--')
plt.xticks(rotation=90)
plt.title("Forecast de Actualización de Apps")
plt.xlabel("Fecha (Año-Mes)")
plt.ylabel("Número de Apps Actualizadas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Score R^2
print("R^2 Score:", modelo.score(X, y))

# Predecir futuro para los siguientes 3 meses
futuro = np.array([len(apps_por_mes), len(apps_por_mes)+1, len(apps_por_mes)+2]).reshape(-1,1)
predicciones_futuras = modelo.predict(futuro)

print("\nPredicción de número de actualizaciones para los siguientes 3 meses:")
print(predicciones_futuras)
