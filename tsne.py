#%%
import sklearn as skl
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import plotly.express as px
import pandas as pd


X, y = load_digits(return_X_y=True)

# %% t-SNE 2D projection

tsne2 = TSNE(n_components=2)
%time X_embedded_2 = tsne2.fit_transform(X)
df2 = pd.DataFrame()
df2['x'] = X_embedded_2[:,0]
df2['y'] = X_embedded_2[:,1]
df2['cluster'] = y

%time fig = px.scatter(df2, x='x', y='y',  color='cluster',   opacity=0.7, size_max=.01)
fig.show()

# %% t-SNE 3D projection 
tsne = TSNE(n_components=3)
%time X_embedded = tsne.fit_transform(X)
df = pd.DataFrame()
df['x'] = X_embedded[:,0]
df['y'] = X_embedded[:,1]
df['z'] = X_embedded[:,2]
df['cluster'] = y

%time  fig = px.scatter_3d(df, x='x', y='y', z='z',color='cluster', opacity=0.7, size_max=.01)
fig.show()

# %%
