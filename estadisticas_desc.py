import pandas as pd

df = pd.read_csv('googleplaystore_limpio.csv')

"""
print(df.describe())

print(df.info())

print(df.head())

print(df.columns)

"""
#Entidades agrupadas

#Apps
grup_app=df.groupby("App").agg({
    "Installs":["mean"],
    "Rating":["mean"],
    "Price":[lambda x: (x == 0).sum(), lambda x: (x>0).sum() ]
}).reset_index()

grup_app.columns=["App", "Total Installs", "Rating", "Gratis", "Paga"]

print("\n\n\n", grup_app)

#Categorias
grup_cat=df.groupby("Category").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"],
    "Price":[lambda x: (x == 0).sum(), lambda x: (x>0).sum() ]
}).reset_index()

grup_cat.columns=["Categoria", "Rating Prom", "Apps", "Total Installs", "Gratis", "Paga"]

print("\n\n\n", grup_cat)

#Generos
grup_gen=df.groupby("Genres").agg({
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

grup_gen.columns=["Generos", "Rating Prom", "Apps", "Total Installs"]

print("\n\n\n", grup_gen)

#Current version
grup_cv=df.groupby("Current Ver").agg({
    "Category":["size"],
    "Rating":["mean"],
    "App":["size"],
    "Installs":["sum"]
}).reset_index()

print("\n\n\n", grup_cv)