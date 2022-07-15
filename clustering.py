from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import scipy.sparse as sps
from math import sqrt
from logger import LoggerFactory
logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')


#https://jtemporal.com/kmeans-and-elbow-method/
def optimal_number_of_clusters(wcss, half):
    x1, y1 = 2, wcss[0]
    x2, y2 = half, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)

    return distances.index(max(distances)) + 1


def kmeans(X, k, rows):
    if k < 1:
        logger.info(f'The K value passed {k} is not valid. Trying to discover the best K automatically.')
        wcss = []
        half = int(rows / 2)
        for i in range(1, half):
            logger.debug(f'Processing loop {i} of {half} in K-Means discovery method.')
            km = KMeans(n_clusters=i, random_state=42, verbose=1)
            km.fit(X)
            wcss.append(km.inertia_)
        optimal = optimal_number_of_clusters(wcss, half)
        logger.info(f'The optimal number of clusters for this dataset is {optimal}.')
        return KMeans(n_clusters=optimal, random_state=42).fit_predict(X)

    return KMeans(n_clusters=k, random_state=42).fit_predict(X)


def dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def agglomerative(X, k):
    return AgglomerativeClustering(n_clusters=k).fit_predict(X.toarray() if sps.isspmatrix(X) else X)


def gmm(X, k):
    return GaussianMixture(n_components=k, random_state=42).fit_predict(X.toarray() if sps.isspmatrix(X) else X)
