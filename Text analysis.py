import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('googleplaystore_limpio.csv')

# Usar columna de texto (nombres de las apps)
texto1 = ' '.join(df['App'].dropna().astype(str).unique())

# Usar columna de texto (nombres de las apps)
texto2 = ' '.join(df['Category'].dropna().astype(str).unique())

# Crear la nube de palabras de apps
wordcloud1 = WordCloud(
    width=1000, 
    height=600, 
    background_color='white', 
    colormap='viridis', 
    max_words=200
).generate(texto1)

# Crear la nube de palabras de categoria
wordcloud2 = WordCloud(
    width=1000, 
    height=600, 
    background_color='white', 
    colormap='viridis', 
    max_words=200
).generate(texto2)

# Mostrar la nube de apps
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Aplicaciones", fontsize=16)
plt.show()

# Mostrar la nube de categorias
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Categorias", fontsize=16)
plt.show()
