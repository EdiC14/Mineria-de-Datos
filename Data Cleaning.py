import pandas as pd

df = pd.read_csv("googleplaystore.csv")  

print(df.info())

print(df.columns)

print(df.dtypes)

print("Número de filas:", len(df))

df = df.drop_duplicates()

df = df.dropna()

df['Last Updated'] = pd.to_datetime(df['Last Updated'])

df['Reviews'] = pd.to_numeric(df['Reviews'])

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(float)

df['Price'] = df['Price'].str.replace('$', '', regex=False).astype(float)

# Guardar el dataset limpio en un nuevo archivo CSV
df.to_csv("googleplaystore_clean.csv", index=False)
