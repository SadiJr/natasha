import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def pca(X, labels=None, texts=None, title='Grupos encontrados - PCA', d=False):
    if d:
        pca = PCA(n_components=3, random_state=42).fit_transform(X.toarray())
        return px.scatter_3d(pca, x=0, y=1, z=2, color=labels, labels=labels, title=title, hover_name=texts)

    pca = PCA(random_state=42).fit_transform(X.toarray())
    return px.scatter(pca, x=0, y=1, color=labels, title=title, hover_name=texts)


def tsne(X, labels, texts, title='Grupos encontrados - TSNE', d=False):
    if d:
        tsne = TSNE(n_components=3, random_state=42).fit_transform(X)
        return px.scatter_3d(tsne, x=0, y=1, z=2, color=labels, labels=labels, title=title, hover_name=texts)

    tsne = TSNE(random_state=42).fit_transform(X)
    return px.scatter(tsne, x=0, y=1, color=labels, labels=labels, title=title, hover_name=texts)
