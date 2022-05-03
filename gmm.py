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
from sklearn.mixture import GaussianMixture
import numpy as np


def main():
    logger.info(f'Starting GaussianMixture')

    df = utils.read_data('./data/concat.txt', None)

    start = timer()
    df = clean_text(df)
    end = timer()
    logger.info(f'Time to clean text: {timedelta(seconds=end - start)}')

    start = timer()
    X, terms = tfidf.tfidf(df.clean, 5, 0.95)
    end = timer()
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}')

    loop(df, X, terms)


def clean_text(df):
    logger.info(f'Cleaning dataset.')
    df['clean'] = df.text.apply(lambda x: preprocess.preprocess(x, True, True))
    logger.info(f'Dataset clean text sample: {df.head()}')
    return df


def loop(df, X, terms):
    results = pd.DataFrame()
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    tolerances = np.arange(start=0.01, stop=0.1, step=0.01)
    results = pd.DataFrame()
    for i in range(10, 20):
        for cov in covariance_type:
            for tol in tolerances:
                results = results.append(gmm(df, X, terms, i, cov, tol))

    results.to_csv('./gmm_loop_result.csv', index=False)


def gmm(df, X, terms, K, cov, tolerance):
    start = timer()

    gmm = GaussianMixture(n_components=K, covariance_type=cov, tol=tolerance)
    X_t = gmm.fit_predict(X.toarray())
    end = timer()

    description = f'Metrics of GMM with {K}, covariance_type {cov} and tolerance {tolerance} are: '
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

    try_result = pd.DataFrame(data=[[K, cov, tolerance, silhouette_euclidian, silhouette_cosine,
                                     silhouette_manhattan, calinski, davies, duration]],
                              columns=['K', 'covariance_type', 'tol', 'silhouette_euclidian', 'silhouette_cosine',
                                       'silhouette_manhattan', 'calinski', 'davies', 'time'])
    logger.info(f'Saving 2d and 3d plots.')
    name = str(gmm)
    text =f"""- {description}
    {silhouette_score_euclidian}
    {silhouette_score_cosine}
    {silhouette_score_manhattan}
    {calinski_harabasz_score}
    {davies_bouldin_score}
    {total_time}"""

    utils.plot2d('./gmm/figures/2d', name, X, X_t, True, footnote=text)
    utils.plot3d('./gmm/figures/3d', name, X, X_t, True, footnote=text)

    df.to_csv(f'./gmm/csvs/{K}-{cov}-{tolerance}.csv')
    pickle.dump(gmm, open(f'./gmm/models/{K}-{cov}-{tolerance}.pkl', 'wb'))

    return try_result


if __name__ == '__main__':
    logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")

    logger.info('Starting GaussianMixture')
    main()

