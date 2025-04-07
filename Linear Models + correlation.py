import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv('googleplaystore_limpio.csv')

#Entidades agrupadas

#Apps
grup_app=df.groupby("App").agg({
    "Installs":["mean"],
    "Rating":["mean"],
    "Price":[lambda x: (x == 0).sum(), lambda x: (x>0).sum() ]
}).reset_index()

grup_app.columns=["App", "Total Installs", "Rating", "Gratis", "Paga"]  

#print("\n\n\n", grup_app)

#Categorias
grup_cat=df.groupby("Category").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"],
    "Price":[lambda x: (x == 0).sum(), lambda x: (x>0).sum() ]
}).reset_index()

grup_cat.columns=["Categoria", "Rating Prom", "Apps", "Total Installs", "Gratis", "Paga"]

#print("\n\n\n", grup_cat)

#Generos
grup_gen=df.groupby("Genres").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

grup_gen.columns=["Generos", "Rating Prom", "Apps", "Total Installs"]

#print("\n\n\n", grup_gen)

#Current version
grup_cv=df.groupby("Current Ver").agg({
    "Category":["size"],
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

# Datos
X = grup_cat[["Total Installs"]]  # Variable independiente
y = grup_cat["Rating Prom"]       # Variable dependiente

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir con el modelo
y_pred = modelo.predict(X_test)

# Calcular R2
r2 = r2_score(y_test, y_pred)
print(f"\nR2 del modelo: {r2:.4f}")

# Visualización
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test["Total Installs"], y=y_test, label="Datos reales", color="blue")
sns.lineplot(x=X_test["Total Installs"], y=y_pred, label="Modelo lineal", color="red")
plt.title("Modelo Lineal: Rating vs Total Installs")
plt.xlabel("Total Installs")
plt.ylabel("Rating Prom")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
