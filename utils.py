from logger import LoggerFactory
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from wordcloud import WordCloud
from os.path import sep
from numpy import unique
import os
import traceback
import sys
from sklearn import metrics
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import re

logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')


def read_data(path, header=None, rename_column=True, text_column=0, text_column_name='text', drop_other_columns=True):
    logger.info(f'Reading dataset at {path}.')

    if not os.path.isfile(path):
        msg = f'Cannot find any file at path {path}.'
        logger.error(msg)
        raise Exception(msg)

    df = pd.read_csv(path, header=header)
    column_to_keep = text_column
    if rename_column:
        df.rename(columns={text_column: text_column_name}, inplace=True)
        column_to_keep = text_column_name

    if drop_other_columns:
        all_columns = list(df.columns)
        all_columns.remove(column_to_keep)
        df.drop(all_columns, axis=1, inplace=True)

    logger.info(f'Dataset have {df.shape[0]} rows.')
    logger.info(f'Dataset info: {df.info()}.')
    logger.info(f'Dataset sample: {df.head()}.')

    return df


def pca_reduction(X, components, random_state):
    logger.debug(f'Applying PCA reduction in dataset using n_components {components} and random state {random_state}')
    pca = PCA(n_components=components, random_state=random_state)
    return pca.fit_transform(X)


def tsne_reduction(X, n_components, random_state):
    logger.debug(f'Applying TSNE reduction in dataset using  random state {random_state}.')
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(X)


def d2(name, labels, pca, tsne, output_dir, footnote=None):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    title = f'Number of clusters: {len(unique(labels))}'
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    if footnote is not None:
        plt.figtext(0.05, 0.01, footnote, fontsize=8, va='bottom', ha='left')

    ax.scatter(pca[:, 0], pca[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
    plt.title(title)
    plt.savefig(f'{output_dir}{name}-pca.png')
    plt.close('all')

    ax.scatter(tsne[:, 0], tsne[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
    plt.title(title)
    plt.savefig(f'{output_dir}{name}-tsne.png')
    plt.close('all')


def d3(name, labels, pca, tsne, output_dir, footnote=None):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    title = f'Number of clusters: {len(unique(labels))}'

    if footnote is not None:
        plt.figtext(0.05, 0.01, footnote, fontsize=8, va='bottom', ha='left')

    ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=labels, edgecolor='k', s=50, alpha=0.5)
    plt.title(title)
    plt.savefig(f'{output_dir}{name}-pca.png')
    plt.close('all')

    ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=labels, edgecolor='k', s=50, alpha=0.5)
    plt.title(title)
    plt.savefig(f'{output_dir}{name}-tsne.png')
    plt.close('all')


# https://stackoverflow.com/a/47247159
def full_stack():
    exc = sys.exc_info()[0]
    if exc is not None:
        f = sys.exc_info()[-1].tb_frame.f_back
        stack = traceback.extract_stack(f)
    else:
        stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
        stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


def calc_metrics(X, X_t, reduction_type, start, end):
    silhouette_euclidian = metrics.silhouette_score(X, X_t, metric='euclidean')
    silhouette_cosine = metrics.silhouette_score(X, X_t, metric='cosine')
    silhouette_manhattan = metrics.silhouette_score(X, X_t, metric='manhattan')

    calinski = metrics.calinski_harabasz_score(X.toarray() if reduction_type is None else X, X_t)
    davies = metrics.davies_bouldin_score(X.toarray() if reduction_type is None else X, X_t)

    duration = timedelta(seconds=end - start)

    return silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration


def log_metrics(description, silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration):
    silhouette_euclidian_description = f'Silhouette Coefficient Euclidian: {silhouette_euclidian}'
    silhouette_cosine_description = f'Silhouette Coefficient Cosine: {silhouette_cosine}'
    silhouette_manhattan_description = f'Silhouette Coefficient Manhattan: {silhouette_manhattan}'

    calinski_description = f'Calinski-Harabasz Index: {calinski}'
    davies_description = f'Davies-Bouldin Index: {davies}'

    total_time = f'Total execution time: {duration}'

    text = f"""- {description}
            {silhouette_euclidian_description}
            {silhouette_cosine_description}
            {silhouette_manhattan_description}
            {calinski_description}
            {davies_description}
            {total_time}"""

    return text


def word_cloud(tokens, title='Cluster top terms:', save=False, path='/tmp/', name='cloud', footnote=None):
    words = ' '.join(tokens) + ' '
    wordcloud = WordCloud(width=800, height=800, background_color='white',
                          min_font_size=10).generate(words)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)

    if footnote:
        plt.figtext(.5, .05, footnote, fontsize=8, ha='center')

    if save:
        if not path.endswith(sep):
            path += sep
        if not name.endswith(sep):
            name += sep

        full_path = f'{path}{name}'
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        plt.savefig(f'{full_path}.png')
        logger.info(f'Figure saved at {full_path}.png.')
        plt.close('all')
    else:
        plt.show()


def check_user_story_pattern(x):
    x = re.sub('\s\s+', ' ', x)
    x = x.strip()
    matchs = re.match('^As (an|a) [\w\W]*, I [\w\W]*, so [\w\W]+\.$', x)
    return matchs is not None


def discard_wrong_user_stories(df):
    df['correctly'] = df.text.apply(lambda x: check_user_story_pattern(x))
    to_process = df.loc[df.correctly]
    to_process.reset_index(inplace=True)
    return to_process


def count_contractions(texts):
    return texts.str.count("'").sum()


# https://stackoverflow.com/a/27084708
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def clean(x):
    x = x.lower()
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
    x = re.sub("\'", '', x)
    x = re.sub('\s\s+' , ' ', x)
    return x.strip()
