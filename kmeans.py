#!/usr/bin/env python
# coding: utf-8

import pickle
from timeit import default_timer as timer
from datetime import timedelta

import preprocess
import tfidf
import utils
from loggerfactory import LoggerFactory
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans


def main():
    logger.info(f'Starting Kmeans')

    df = utils.read_data('./data/concat.txt', None)

    start = timer()
    df = clean_text(df)
    end = timer()
    logger.info(f'Time to clean text: {timedelta(seconds=end - start)}')

    start = timer()
    X, terms = tfidf.tfidf(df.clean, 5, 0.95)
    end = timer()
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}')

    km_loop(df, X, terms)


def clean_text(df):
    logger.info(f'Cleaning dataset.')
    df['clean'] = df.text.apply(lambda x: preprocess.preprocess(x, True, True))
    logger.info(f'Dataset clean text sample: {df.head()}')
    return df


def km_loop(df, X, terms):
    results = pd.DataFrame()
    for i in range(10, 20):
        for j in range(100, 110):
            for k in range(10, 20):
                results = results.append(kmeans(df, X, terms, i, j, k))

    results.to_csv('./Kmeans_loop_result.csv', index=False)


def kmeans(df, X, terms, K, max_iter, n_init):
    start = timer()

    km = KMeans(n_clusters=K, max_iter=max_iter, n_init=n_init, random_state=61659)
    X_t = km.fit_predict(X)
    end = timer()

    description = f'Metrics of KMeans with {K}, max_iter {max_iter} and n_init {n_init} are: '
    calinski = metrics.calinski_harabasz_score(X.toarray(), X_t)
    davies = metrics.davies_bouldin_score(X.toarray(), X_t)
    silhouette_euclidian = metrics.silhouette_score(X, X_t, metric='euclidean')
    silhouette_cosine = metrics.silhouette_score(X, X_t, metric='cosine')
    silhouette_manhattan = metrics.silhouette_score(X, X_t, metric='manhattan')
    duration = timedelta(seconds=end - start)

    calinski_harabasz_score = f'Calinski-Harabasz Index: {calinski}'
    davies_bouldin_score = f'Davies-Bouldin Index: {davies}'
    silhouette_score_euclidian = f'Silhouette Coefficient Euclidian: {silhouette_euclidian}'
    silhouette_score_cosine = f'Silhouette Coefficient Cosine: {silhouette_cosine}'
    silhouette_score_manhattan = f'Silhouette Coefficient Manhattan: {silhouette_manhattan}'
    total_time = f'Total execution time: {duration}'

    logger.debug(description)
    logger.debug(silhouette_score_euclidian)
    logger.debug(silhouette_score_cosine)
    logger.debug(silhouette_score_manhattan)
    logger.debug(calinski_harabasz_score)
    logger.debug(davies_bouldin_score)
    logger.debug(total_time)
    df['cluster'] = X_t

    try_result = pd.DataFrame(data=[[K, max_iter, n_init, silhouette_euclidian, silhouette_cosine,
                                     silhouette_manhattan, calinski, davies, duration]],
                              columns=['K', 'max_iter', 'n_init', 'silhouette_euclidian', 'silhouette_cosine',
                                       'silhouette_manhattan', 'calinski', 'davies', 'time'])
    centroids = km.cluster_centers_.argsort()[:, ::-1]

    logger.info(f'Saving 2d and 3d plots.')
    name = str(km)
    text =f"""- {description}
    {silhouette_score_euclidian}
    {silhouette_score_cosine}
    {silhouette_score_manhattan}
    {calinski_harabasz_score}
    {davies_bouldin_score}
    {total_time}"""

    utils.plot2d('./kmeans/figures/2d', name, X, X_t, True, centroids=km.cluster_centers_, footnote=text)
    utils.plot3d('./kmeans/figures/3d', name, X, X_t, True, centroids=km.cluster_centers_, footnote=text)
    for i in range(K):
        utils.word_cloud('./kmeans/figures/wordcloud', name, i, terms[centroids[i, :10]], True, footnote=text)

    df.to_csv(f'./kmeans/csvs/{K}-{max_iter}-{n_init}.csv')
    pickle.dump(km, open(f'./kmeans/models/{K}-{max_iter}-{n_init}.pkl', 'wb'))

    return try_result


if __name__ == '__main__':
    logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")

    logger.info('Starting Kmeans')
    main()

