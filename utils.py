from loggerfactory import LoggerFactory
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from wordcloud import WordCloud
from os.path import sep
from numpy import unique
import os

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


def read_data(path, header):
    logger.info(f'Reading dataset {path}')

    df = pd.read_csv(path, header=header)
    df.rename(columns = {0:'text'}, inplace = True)
    df.drop(1, axis=1, inplace=True)

    logger.info(f'Dataset have {df.shape[0]} rows.')
    logger.info(f'Dataset info: {df.info()}')
    logger.info(f'Dataset sample: {df.head()}')

    return df


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


def word_cloud(path, name, cluster, tokens, save, footnote=None):
    words = ' '.join(tokens) + ' '
    wordcloud = WordCloud(width=800, height=800, background_color='white',
                          min_font_size=10).generate(words)
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {cluster} top terms:')
    if footnote:
        plt.figtext(.5, .05, footnote, fontsize=8, ha='center')

    if save:
        if not path.endswith(sep):
            path += sep
        if not name.endswith(sep):
            name += sep

        full_path = f'{path}{name}'
        if not os.path.isdir(full_path):
            os.mkdir(full_path)

        plt.savefig(f'{path}{name}{cluster}.png')
        plt.close('all')
    else:
        plt.show()