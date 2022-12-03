import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.preprocessing import StandardScaler
from model.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__, 'DEBUG')


def _str_to_bool(inp):
    return inp == "True"


class DimReduction:
    _default_options = {}

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class PCA(DimReduction):
    _default_options = {'n_components': 2}

    def __init__(self, n_components=2):
        self.options = {"n_components": float(n_components) if n_components.count('.') > 0 else int(n_components)}

    def apply(self, df):
        #logger.debug(f'df:\n{df}')
        arr = df.values
        pca = decomposition.PCA(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}, random_state=42)
        logger.debug(f'PCA definition: {pca}')
        return pca.fit_transform(arr)


class TSNE(DimReduction):
    _default_options = {'n_components': 2}

    def __init__(self, n_components=2):
        self.options = {'n_components': int(n_components)}

    def apply(self, df):
        #logger.debug(f'df: {df}')
        arr = df.values
        tsne = manifold.TSNE(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}, random_state=42)
        logger.debug(f'TSNE definition: {tsne}.')
        return tsne.fit_transform(arr)

