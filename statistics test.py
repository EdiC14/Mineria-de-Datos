from scipy import stats

import pandas as pd

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

#Variable a comparar por Categoria

categorias=grup_cat["Categoria"].unique()

grupos=[grup_cat[grup_cat["Categoria"]==cat]["Rating Prom"].dropna() for cat in categorias]

#Prueba de normalidad con Shapiro-Wilk

print("\n\n")

for i, grupo in enumerate(grupos):
    stat, p=stats.shapiro(grupo)
    print(f"Shapiro-Wilk para {categorias[i]}: p = {p:.5f}")

#Se aplica ANOVA si los datos son normales

anova_stat, anova_p=stats.f_oneway(*grupos)

print(f"\nANOVA: p = {anova_p:.5f}")

#Se aplica Kruskal-Wallis si los datos no son normales

kruskal_stat, kruskal_p=stats.kruskal(*grupos)

print(f"\nKruskal-Wallis: p = {kruskal_p:.5f}")