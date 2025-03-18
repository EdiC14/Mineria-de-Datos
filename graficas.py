import pandas as pd

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

grup_app.columns=["App", "Total Instalaciones", "Rating", "Gratis", "Paga"]

print("\n\n\n", grup_app)

#Categorias
grup_cat=df.groupby("Category").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"],
    "Price":[lambda x: (x == 0).sum(), lambda x: (x>0).sum() ]
}).reset_index()

grup_cat.columns=["Categoria", "Rating Prom", "Apps", "Total Instalaciones", "Gratis", "Paga"]

print("\n\n\n", grup_cat)

#Generos
grup_gen=df.groupby("Genres").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

grup_gen.columns=["Generos", "Rating Prom", "Apps", "Total Instalaciones"]

print("\n\n\n", grup_gen)

#Current version
grup_cv=df.groupby("Current Ver").agg({
    "Category":["size"],
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

grup_cv.columns=["Version Resiente","Categorias", "Rating Prom", "Apps", "Total Instalaciones"]

print("\n\n\n", grup_cv)

#Graficas
graficos = [
    {"tipo": "bar", "df": grup_cat, "x": "Categoria", "y": "Rating Prom","titulo": "Promedio de Rating por Categoría"},
    {"tipo": "bar", "df": grup_cat, "x": "Categoria", "y": "Apps", "titulo": "Total de Aplicaciones por Categoría"},
    {"tipo": "pie", "df": grup_cat, "columna": "Categoria", "valor": "Apps", "titulo": "Distribución de Apps por Categoría"},
    {"tipo": "hist", "df": grup_app, "columna": "Rating", "titulo": "Histograma de Ratings"},
    {"tipo": "box", "df": grup_app, "columna": "Total Instalaciones", "titulo": "Boxplot de Instalaciones por App"},
    {"tipo": "scatter", "df": grup_app, "x": "Rating", "y": "Total Instalaciones", "titulo": "Relación entre Rating e Instalaciones"}
]

for graf in graficos:
    plt.figure(figsize=(10, 5))

    if graf["tipo"] == "bar":
        sns.barplot(data=graf["df"], x=graf["x"], y=graf["y"], palette="coolwarm")
        plt.xticks(rotation=90)
    
    elif graf["tipo"] == "pie":
        data_pie = graf["df"].nlargest(10, graf[ "valor"])  # Toma las 10 categorías con más apps
        plt.pie(data_pie[graf["valor"]], labels=data_pie[graf["columna"]], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    
    elif graf["tipo"] == "hist":
        sns.histplot(graf["df"][graf["columna"]], bins=20, kde=True, color="purple")
    
    elif graf["tipo"] == "box":
        sns.boxplot(x=graf["df"][graf["columna"]])
    
    elif graf["tipo"] == "scatter":
        sns.scatterplot(x=graf["df"][graf["x"]], y=graf["df"][graf["y"]], alpha=0.5, color="red")
    
    plt.title(graf["titulo"])
    plt.show()