import os

from model.logger import LoggerFactory
from model import preprocess

logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')

_path = os.path.dirname(__file__)


def get_demo_data(filename='demo.csv'):
    path = os.path.join(_path, filename)

    logger.debug(f'Trying to read data source: {path}.')
    df = preprocess.read_user_stories(path)

    #logger.debug(f'Readed dataframe:\n{df}')
    return df
