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
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from numpy import unique


# In[ ]:

def main():
    logger.info(f'Starting DBSCAN')

    df = utils.read_data('./data/concat.txt', None)

    start = timer()
    df = clean_text(df)
    end = timer()
    logger.info(f'Time to clean text: {timedelta(seconds=end - start)}')

    start = timer()
    X, terms = tfidf.tfidf(df.clean, 5, 0.95)
    end = timer()
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}')

    dbscan_loop(df, X, terms)


def clean_text(df):
    logger.info(f'Cleaning dataset.')
    df['clean'] = df.text.apply(lambda x: preprocess.preprocess(x, True, True))
    logger.info(f'Dataset clean text sample: {df.head()}')
    return df


def dbscan_loop(df, X, terms):
    results = pd.DataFrame()
    eps_list = np.arange(start=0.1, stop=10, step=0.1)
    min_samples_list = np.arange(start=5, stop=50, step=1)
    metrics_try = ['cosine', 'euclidean', 'manhattan']

    for metric in metrics_try:
        for eps in eps_list:
            for sample in min_samples_list:
                results = results.concat(dbscan(df, X, terms, metric, eps, sample))

    results.to_csv('./dbscan_loop_result.csv', index=False)


def dbscan(df, X, terms, metric, eps, min_sample):
    start = timer()
    dbscan = DBSCAN(metric=metric, eps=eps, min_samples=min_sample)
    X_t = dbscan.fit_predict(X)
    end = timer()

    if len(unique(X_t)) > 1:
        description = f'Metrics of DBSCAN with metric {metric}, EPS {eps} and min samples {min_sample} are: '

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

        try_result = pd.DataFrame(data=[[metric, eps, min_sample, silhouette_euclidian, silhouette_cosine,
                                         silhouette_manhattan, calinski, davies, duration]],
                                  columns=['metric', 'EPS', 'min_samples',  'silhouette_euclidian', 'silhouette_cosine',
                                           'silhouette_manhattan', 'calinski', 'davies', 'time'])

        logger.info(f'Saving 2d and 3d plots.')
        name = str(dbscan)
        text =f"""- {description}
           {silhouette_score_euclidian}
            {silhouette_score_cosine}
            {silhouette_score_manhattan}
            {calinski_harabasz_score}
            {davies_bouldin_score}
            {total_time}"""

        utils.plot2d('./dbscan/figures/2d', name, X, X_t, True, footnote=text)
        utils.plot3d('./dbscan/figures/3d', name, X, X_t, True, footnote=text)

        df.to_csv(f'./dbscan/csvs/{metric}-{eps}-{min_sample}.csv')
        pickle.dump(dbscan, open(f'./dbscan/models/{metric}-{eps}-{min_sample}.pkl', 'wb'))

        return try_result


if __name__ == '__main__':
    logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")

    logger.info('Starting DBSCAN')
    main()

