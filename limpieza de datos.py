import pandas as pd

df = pd.read_csv('googleplaystore.csv')

print(df.head()) #los primeros 5

print("\n\n", df.info())

print("\n\n", df.columns)

df.dropna(inplace=True) #se borran filas con datos faltantes

print("\n\n", df.info())

df.drop_duplicates() #borra duplicados en general

print("\n\n", df.info())

print(df.shape) #ver filas y columnas

df['Installs']=df['Installs'].str.replace('[+,]','',regex=True).astype(float)

df['Price']=df['Price'].str.replace('[$,]', '',regex=True).astype(float)

df['Last Updated']=pd.to_datetime(df['Last Updated'])

df['Reviews']=pd.to_numeric(df['Reviews'])

df.to_csv("googleplaystore_limpio.csv", index=False)