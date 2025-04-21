import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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

#print("\n\n\n", grup_cv)

#Crear columna de clasificación (rating alto o bajo)
media_rating = grup_cat["Rating Prom"].mean()
grup_cat["High_Rating"] = grup_cat["Rating Prom"].apply(lambda x: 1 if x >= media_rating else 0)

#Definir variables
X = grup_cat[["Total Installs", "Apps"]]  # Variables numéricas para entrenar
y = grup_cat["High_Rating"]               # Variable objetivo

#Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crear modelo KNN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Predecir
y_pred = knn.predict(X_test)

#Evaluar el modelo
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))

#Visualizar la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión - KNN")
plt.show()
