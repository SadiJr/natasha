from loggerfactory import LoggerFactory
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from wordcloud import WordCloud
from os.path import sep
from numpy import unique
import os
import traceback
import sys
import argparse
from sklearn import metrics
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')


def elbow_plot(min_K, max_K, max_iter, n_init, X, save):
    distortions = []
    loop = range(min_K, max_K)
    for i in loop:
        km = KMeans(n_clusters=i, max_iter=max_iter, n_init=n_init)
        distortions.append(km.fit(X).inertia_)

    plt.figure(figsize=(15, 5))
    plt.plot(loop, distortions)
    plt.grid()

    if save:
        plt.savefig(f'./elbows/kmeans-{min_K}-{max_K}-{max_iter}-{n_init}.png')
        plt.close('all')
    else:
        plt.grid()


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
    Y = pca.fit_transform(X)
    return pca, Y


def tsne_reduction(X, components, random_state):
    logger.debug(f'Applying TSNE reduction in dataset using n_components {components} and random state {random_state}')
    tsne = TSNE(n_components=components, random_state=random_state)
    Y = tsne.fit_transform(X)
    return tsne, Y


def d2(name, X, labels, original, random_state, output_dir, footnote=None):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    title = f'Number of clusters: {len(unique(labels))}'
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    if footnote is not None:
        plt.figtext(0.05, 0.01, footnote, fontsize=8, va='bottom', ha='left')

    if not original:
        ax.scatter(X[:, 0], X[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}.png')
        plt.close('all')
    else:
        logger.info(f'The dataset to plot is the original sparse matrix. Need to reduct to plot. '
                    f'Trying with PCA and TSNE using n_components 2 and random_state {random_state}.')
        pca = PCA(n_components=2, random_state=random_state).fit_transform(X.toarray())
        tsne = TSNE(n_components=2, random_state=random_state).fit_transform(X)

        ax.scatter(pca[:, 0], pca[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}-pca-{random_state}.png')
        plt.close('all')

        ax.scatter(tsne[:, 0], tsne[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}-tsne-{random_state}.png')
        plt.close('all')


def d3(name, X, labels, original, random_state, output_dir, footnote=None):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    title = f'Number of clusters: {len(unique(labels))}'

    if footnote is not None:
        plt.figtext(0.05, 0.01, footnote, fontsize=8, va='bottom', ha='left')

    if not original:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}.png')
        plt.close('all')
    else:
        logger.info(f'The dataset to plot is the original sparse matrix. Need to reduce to plot. '
                    f'Trying with PCA and TSNE using n_components 3 and random_state {random_state}.')
        pca = PCA(n_components=3, random_state=random_state).fit_transform(X.toarray())
        tsne = TSNE(n_components=3, random_state=random_state).fit_transform(X)

        ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}-pca-{random_state}.png')
        plt.close('all')

        ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=labels, edgecolor='k', s=50, alpha=0.5)
        plt.title(title)
        plt.savefig(f'{output_dir}{name}-tsne-{random_state}.png')
        plt.close('all')


def plot2d(path, name, X, labels, save, centroids=None, footnote=None):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    pca = PCA(n_components=2, random_state=61659)
    clusters = pca.fit_transform(X.toarray())

    ax.scatter(clusters[:, 0], clusters[:, 1], c=labels, edgecolor='k', s=50, alpha=0.5)
    if centroids is not None:
        cluster_centroids = pca.transform(centroids)
        for i, c in enumerate(cluster_centroids):
            ax.scatter(c[0], c[1], marker=f'${i}$', s=300, c='r', label='Centroid')

    plt.title(f'Number of clusters: {len(unique(labels))}')

    if footnote:
        plt.figtext(0.05, 0.01, footnote, fontsize=8, va='bottom', ha='left')

    if save:
        if not path.endswith(sep):
            path += sep
        plt.savefig(f'{path}{name}.png')
        plt.close('all')
    else:
        plt.show()


def plot3d(path, name, X, labels, save, centroids=None, footnote=None):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    pca = PCA(n_components=3, random_state=61659)
    clusters = pca.fit_transform(X.toarray())

    ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], c=labels,
               edgecolor='k', s=50, alpha=0.5)

    if centroids is not None:
        cluster_centroids = pca.transform(centroids)
        for i, c in enumerate(cluster_centroids):
            ax.scatter(c[0], c[1], marker=f'${i}$', s=300, c='r', label='Centroid')

    plt.title(f'Number of clusters: {len(unique(labels))}')

    if footnote:
        fig.text(.5, .05, footnote, fontsize=8, ha='center')

    if save:
        if not path.endswith(sep):
            path += sep
        plt.savefig(f'{path}{name}.png')
        plt.close('all')
    else:
        plt.show()


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


def default_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', help='the dataset input file', type=str, default='./data/stories.csv')
    parser.add_argument('--head', help='indicates if the dataset have or not a head', default=None, action='store_true')
    parser.add_argument('--stop_words', help='indicates to not use stop words to clean text string', default=True,
                        action='store_false')
    parser.add_argument('--lem', help='indicates to not use lemmatizer to clean text string', default=True,
                        action='store_false')
    parser.add_argument('--stem', help='indicates to not use stemmer to clean text string', default=True,
                        action='store_false')
    parser.add_argument('--min-df', help='ignore terms that have a document frequency strictly lower than the given '
                                         'threshold', type=int, default=5)
    parser.add_argument('--max-df', help='ignore terms that have a document frequency strictly higher than the given '
                                         'threshold if the dataset have or not a head', type=float, default=0.95)
    parser.add_argument('--matrix', help='the matrix who will be used to cluster data', type=str, default='all',
                        choices=['all', 'original', 'pca', 'tsne'])
    parser.add_argument('--n-components', help='the number of components to make the matrix reduction', type=int,
                        default=2)
    parser.add_argument('--random-state', help='the random state used to made the matrix reduction', type=int,
                        default=11767)
    parser.add_argument('--disable-plots', help='disable plots to speed algorithmn loop process', default=False,
                        action='store_true')
    return parser


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
