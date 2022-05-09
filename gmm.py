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
from sklearn.mixture import GaussianMixture
import numpy as np
import tempfile as tmp
import os


default_random_state = None
logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')


def main(dataset, stop_words, lem, stem, min_df, max_df, k_start, k_stop, k_step, tol_start, tol_stop, tol_step,
         covariances, matrix, n_components, random_state, disable_plots, output_dir):
    logger.info(f'Starting GaussianMixture')

    global default_random_state
    default_random_state = random_state

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not output_dir.endswith(os.path.sep):
        output_dir += os.path.sep

    df = utils.read_data(dataset)

    start = timer()
    df = preprocess.clean_text(df, stop_words=stop_words, lemmatize=lem, stem=stem)
    end = timer()
    logger.info(f'Time to clean text: {timedelta(seconds=end - start)}.')

    start = timer()
    X, terms = tfidf.tfidf(df.clean, min_df=min_df, max_df=max_df)
    end = timer()
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}.')

    if matrix == 'original':
        loop(df, X, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, None, output_dir,
             disable_plots)
    elif matrix == 'all':
        pca, Y_pca = utils.pca_reduction(X.toarray(), 0.95, random_state)
        tsne, Y_tsne = utils.tsne_reduction(X, 2, random_state)
        loop(df, X, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, None, output_dir,
             disable_plots)
        loop(df, Y_pca, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, "PCA", output_dir,
             disable_plots)
        loop(df, Y_tsne, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, "TSNE", output_dir,
             disable_plots)
    elif matrix == 'pca':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        loop(df, Y_pca, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, "PCA", output_dir,
             disable_plots)
    else:
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, Y_tsne, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, "TSNE", output_dir,
             disable_plots)


def loop(df, X, covariances, k_start, k_stop, k_step, tol_start, tol_stop, tol_step, reduction_type, output_dir,
         disable_plots):
    if covariances == 'all':
        covariance = ['full', 'tied', 'diag', 'spherical']
    else:
        covariance = [covariances]
    pd.DataFrame(columns=['K', 'covariance_type', 'tol', 'reduction_type', 'silhouette_euclidian', 'silhouette_cosine',
                          'silhouette_manhattan', 'calinski', 'davies', 'time'])\
        .to_csv(f'{output_dir}results-{reduction_type}.csv', index=False)

    for i in np.arange(start=k_start, stop=k_stop, step=k_step):
        for cov in covariance:
            for tol in np.arange(start=tol_start, stop=tol_stop, step=tol_step):
                gmm(df, X, i, cov, tol, reduction_type, output_dir, disable_plots)


def gmm(df, X, K, cov, tolerance, reduction_type, output_dir, disable_plots):
    start = timer()

    gmm = GaussianMixture(n_components=K, covariance_type=cov, tol=tolerance, random_state=default_random_state)
    X_t = gmm.fit_predict(X.toarray() if reduction_type is None else X)
    end = timer()

    description = f'Metrics of GMM with {K}, covariance_type {cov}, tolerance {tolerance} and reduction type ' \
                  f'{reduction_type} are: '
    silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration = \
        utils.calc_metrics(X, X_t, reduction_type, start, end)

    text = utils.log_metrics(description, silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski,
                             davies, duration)
    logger.debug(text)
    df['cluster'] = X_t

    pd.DataFrame(data=[[K, cov, tolerance, reduction_type, silhouette_euclidian, silhouette_cosine,
                        silhouette_manhattan, calinski, davies, duration]],
                 columns=['K', 'covariance_type', 'tol', 'reduction_type', 'silhouette_euclidian', 'silhouette_cosine',
                          'silhouette_manhattan', 'calinski', 'davies', 'time'])\
        .to_csv(f'{output_dir}results-{reduction_type}.csv', index=False, header=False, mode='a')

    if not disable_plots:
        logger.info('Saving 2d and 3d plots.')
        name = str(gmm)

        utils.d2(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}2d',
                 footnote=text)
        utils.d3(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}3d',
                 footnote=text)

    df.to_csv(f'{output_dir}{K}-{cov}-{tolerance}-{reduction_type}.csv', index=False)
    #pickle.dump(gmm, open(f'./gmm/models/{K}-{cov}-{tolerance}.pkl', 'wb'))


def parse_args():
    parser = utils.default_args('Execute GMM clustering in text dataset')

    parser.add_argument('--k-start', help='the minimum number of clusters', type=int, default=2)
    parser.add_argument('--k-stop', help='the maximum number of clusters', type=int, default=30)
    parser.add_argument('--k-step', help='the default step number to increment in the K value', type=int, default=1)
    parser.add_argument('--tolerance-start', help='the default tolerance number to start clustering', type=float,
                        default=0.01)
    parser.add_argument('--tolerance-stop', help='the default tolerance number to stop clustering', type=float,
                        default=1.0)
    parser.add_argument('--tolerance-step', help='the default step number to increment in the tolerance value',
                        type=float, default=0.001)
    parser.add_argument('--covariance-type', help='the covariances to try', type=str, default='all',
                        choices=['all', 'full', 'tied', 'diag', 'spherical'])
    parser.add_argument('--output-dir',  help='the directory to save the generate artifacts', type=str,
                        default=f'{tmp.tempdir}{os.path.sep}gmm')

    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args.dataset, args.stop_words, args.lem, args.stem, args.min_df, args.max_df, args.k_start, args.k_stop,
             args.k_step, args.tol_start, args.tol_stop, args.tol_step, args.covariances, args.matrix,
             args.n_components, args.random_state, args.disable_plots, args.output_dir)
        logger.info('Finish GMM algorithm run')
    except Exception:
        logger.error('Error when running GMM algorithm: ')
        logger.error(utils.full_stack())


