#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# In[2]:


# Cargamos un dataset con estadisticas de todos los jugadores de La Liga de España y contamos cuántos registros tiene el dataset
jugadores = pd.read_csv('data/estadisticas_espana.csv', delimiter=';', header=0, names = ['jugador','equipo','pj','pi','min','gol','asist','tiros','tiro_arco','ataj','vay_inv','tarj_am','tarj_roj','fal_com','fal_rec','offside'])
jugadores.shape


# In[3]:


# Tomamos jugadores que hayan jugado 10 o más partidos
jugadores_10pj = jugadores[jugadores.pj >= 10]


# In[4]:


# Contamos cuantos jugadores quedaron
jugadores_10pj.shape


# In[5]:


# Generamos un arreglo con las dimensiones de interés
X = np.array(jugadores_10pj[["gol", "asist", "tiros", "ataj", "vay_inv", "tarj_am", "fal_com", "fal_rec", "offside"]])


# In[6]:


# Utilizamos el algoritmo KMeans para detectar 4 clusters. 4 clusters porque son las posiciones estandar del futbol: arquero, 
# defensor, volante y delantero.
kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)


# In[7]:


# Vemos la incidencia de cada Dimensión para cada cluster en un gráfico de Radar (hay que descargar el paquete plotly)
import plotly.graph_objects as go

categories = ["gol", "asist", "tiros", "ataj", "vay_inv", "tarj_am", "fal_com", "fal_rec", "offside"]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=centroids[0],
      theta=categories,
      fill='toself',
      name='Posicion A'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[1],
      theta=categories,
      fill='toself',
      name='Posicion B'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[2],
      theta=categories,
      fill='toself',
      name='Posicion C'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[3],
      theta=categories,
      fill='toself',
      name='Posicion D'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()


# In[8]:


# Atajadas y Tiros tienen un rango mayor de valores, por lo que inciden demasiado en el gráfico.
# Utilizamos el zscore para standarizar
X_zscore = StandardScaler().fit_transform(X);


# In[9]:


# Calculamos los clusters luego de aplicar el zscore
kmeans = KMeans(n_clusters=4).fit(X_zscore)
centroids = kmeans.cluster_centers_
print(centroids)


# In[10]:


# Vemos la incidencia de cada dimensión "estandarizada" en un gráfico de radar.
import plotly.graph_objects as go

categories = ["gol", "asist", "tiros", "ataj", "vall_inv", "tarj_am", "fal_com", "fal_rec", "offside"]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=centroids[0],
      theta=categories,
      fill='toself',
      name='Posicion A'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[1],
      theta=categories,
      fill='toself',
      name='Posicion B'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[2],
      theta=categories,
      fill='toself',
      name='Posicion C'
))
fig.add_trace(go.Scatterpolar(
      r=centroids[3],
      theta=categories,
      fill='toself',
      name='Posicion D'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()


# In[11]:


# Importamos TNSE para  reducir dimensiones
from scipy.spatial.distance import pdist
#from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 4)


# In[12]:


# Se asigna directamente el valor de X (con zscore) y se lo transforma a dos dimensiones.
tsne = TSNE()

X_embedded = tsne.fit_transform(X_zscore)


# In[13]:


# Generamos un Dataframe con el indice que identifica a cada jugador y el cluster asignado.
cluster_map = pd.DataFrame()
cluster_map['data_index'] = jugadores_10pj.index.values
cluster_map['cluster'] = kmeans.labels_


# In[14]:


# Graficamos la distribución de los cluster en base a las dos dimensiones principales. 
# Utilizamos el Dataframe generado en el paso anterior para colorear los distintos clusters (parametro hue)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=cluster_map['cluster'] , legend='full', palette=palette)

plt.show()


# In[17]:


#Buscamos jugadores para validar que el cluster asignado es distinto segun la posicion real en que se desempeña cada uno
jugadores_10pj[jugadores_10pj['jugador'].str.contains("Messi")]


# In[18]:


#Vemos que cluster tiene asignado
cluster_map[cluster_map.data_index == 294]


# In[ ]:




