from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import scipy.sparse as sps


def kmeans(X, k):
    return KMeans(n_clusters=k, random_state=42).fit_transform(X)


def dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def agglomerative(X, k):
    return AgglomerativeClustering(n_clusters=k).fit_predict(X.toarray() if sps.isspmatrix(X) else X)


def gmm(X, k):
    return GaussianMixture(n_components=k, random_state=42).fit_predict(X.toarray() if sps.isspmatrix(X) else X)
