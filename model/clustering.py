import sklearn.cluster as sc
import sklearn.mixture as sm
from model import preprocess

from model.logger import LoggerFactory
from sklearn.metrics import pairwise

logger = LoggerFactory.get_logger(__name__, 'DEBUG')


class Clustering:
    _default_options = {}

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class KMeans(Clustering):
    _default_options = {'n_clusters': 8, 'n_init': 10, 'max_iter': 50, 'metric': 'euclidean'}

    def __init__(self, n_clusters, n_init, max_iter, metric):
        self.options = {"n_clusters": int(n_clusters), 'n_init': int(n_init), 'max_iter': int(max_iter), 'metric': metric}

    def apply(self, df):
        logger.debug(f'df: {df.shape}')
        kmeans = sc.KMeans(**{k: v for k, v in self.options.items() if not k.startswith('metric')}, random_state=42)
        logger.debug(f'kmeans: {kmeans}')

        if self.options['metric'] == 'euclidean':
            return kmeans.fit_predict(df.values)
        elif self.options['metric'] == 'cosine':
            return kmeans.fit_predict(pairwise.cosine_distances(df.values))
        elif self.options['metric'] == 'manhattan':
            return kmeans.fit_predict(pairwise.manhattan_distances(df.values))
        else:
            return None


class DBSCAN(Clustering):
    _default_options = {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'}

    def __init__(self, eps, min_samples, metric):
        self.options = {"eps": float(eps), 'min_samples': int(min_samples), 'metric': metric}

    def apply(self, df):
        logger.debug(f'df: {df.shape}')
        dbscan = sc.DBSCAN(**self.options, n_jobs=4)
        logger.debug(f'DBSCAN definition: {dbscan}')
        return dbscan.fit_predict(df.values)


class AgglomerativeClustering(Clustering):
    _default_options = {'n_clusters': 8, 'affinity': 'euclidean', 'linkage': 'ward'}

    def __init__(self, n_clusters, affinity, linkage):
        self.options = {"n_clusters": int(n_clusters), 'affinity': affinity, 'linkage': linkage}

    def apply(self, df):
        logger.debug(f'df: {df.shape}')
        agglo = sc.AgglomerativeClustering(**self.options)
        logger.debug(f'Agglomerative definition: {agglo}')
        arr = df.values
        return agglo.fit_predict(arr)


class GMM(Clustering):
    _default_options = {'n_components': 8, 'covariance_type': 'full', 'tol': 0.001}

    def __init__(self, n_components, covariance_type, tol):
        self.options = {"n_components": int(n_components), 'covariance_type': covariance_type, 'tol': float(tol)}

    def apply(self, df):
        logger.debug(f'df: {df.shape}')
        gmm = sm.GaussianMixture(**self.options, random_state=42)
        logger.debug(f'GMM definition: {gmm}')
        return gmm.fit_predict(df.values)


class Personas(Clustering):
    def __init__(self):
        pass

    def apply(self, df):
        if 'text' in df.columns:
            df, users = preprocess.label_user_types(df)
            df['user_name'] = df['user_type'].apply(lambda x: users[x])
            return df
        return None
