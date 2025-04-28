import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


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

#Variables para clustering
X = grup_cat[["Total Installs", "Apps"]]

#Crear el modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42) 
kmeans.fit(X)

#Asignar los clusters al DataFrame
grup_cat["Cluster"] = kmeans.labels_

#Visualizar resultados
plt.figure(figsize=(10,6))
sns.scatterplot(data=grup_cat, x="Total Installs", y="Apps", hue="Cluster", palette="Set1")
plt.title("Clustering de Categorías de Apps con KMeans")
plt.xlabel("Total Installs")
plt.ylabel("Apps")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

#Ver datos agrupados por cluster
print(grup_cat.groupby("Cluster")[["Rating Prom", "Apps", "Total Installs", "Gratis", "Paga"]].mean())
