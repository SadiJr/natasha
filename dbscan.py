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
import os
from sklearn.cluster import DBSCAN
from numpy import unique
import tempfile as tmp


default_random_state = None
logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")


def main(dataset, stop_words, lem, stem, min_df, max_df, eps_start, eps_stop, eps_step, min_sample_start,
         min_sample_stop, min_sample_step, metrics, matrix, n_components, random_state, output_dir):
    logger.info(f'Starting DBSCAN.')

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
        loop(df, X, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics, None,
             output_dir)
    elif matrix == 'all':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, X, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics, None,
             output_dir)
        loop(df, Y_pca, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics,
             "PCA", output_dir)
        loop(df, Y_tsne, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics,
             "TSNE", output_dir)
    elif matrix == 'pca':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        loop(df, Y_pca, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics,
             "PCA", output_dir)
    else:
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, Y_tsne, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics,
             "TSNE", output_dir)


def loop(df, X, eps_start, eps_stop, eps_step, min_sample_start, min_sample_stop, min_sample_step, metrics,
         reduction_type, output_dir):
    results = pd.DataFrame()
    if metrics == 'all':
        metrics_try = ['cosine', 'euclidean', 'manhattan']
    else:
        metrics_try = [metrics]

    for metric in metrics_try:
        for eps in np.arange(start=eps_start, stop=eps_stop, step=eps_step):
            for sample in np.arange(start=min_sample_start, stop=min_sample_stop, step=min_sample_step):
                results = results.append(dbscan(df, X, metric, eps, sample, reduction_type, output_dir))
    results.to_csv(f'{output_dir}results-{reduction_type}.csv', index=False)


def dbscan(df, X, metric, eps, min_sample, reduction_type, output_dir):
    start = timer()
    dbscan = DBSCAN(metric=metric, eps=eps, min_samples=min_sample)
    X_t = dbscan.fit_predict(X)
    end = timer()

    if len(unique(X_t)) > 1:
        description = f'Metrics of DBSCAN with metric {metric}, EPS {eps}, min samples {min_sample} and reduction ' \
                      f'type {reduction_type} are: '

        silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration = \
            utils.calc_metrics(X, X_t, reduction_type, start, end)

        text = utils.log_metrics(description, silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski,
                                 davies, duration)
        logger.debug(text)
        df['cluster'] = X_t

        try_result = pd.DataFrame(data=[[metric, eps, min_sample, reduction_type, silhouette_euclidian,
                                         silhouette_cosine, silhouette_manhattan, calinski, davies, duration]],
                                  columns=['metric', 'EPS', 'min_samples', 'reduction_type', 'silhouette_euclidian',
                                           'silhouette_cosine', 'silhouette_manhattan', 'calinski', 'davies', 'time'])

        logger.info('Saving 2d and 3d plots.')
        name = str(dbscan)

        utils.d2(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}2d',
                 footnote=text)
        utils.d3(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}3d',
                 footnote=text)

        df.to_csv(f'{output_dir}{metric}-{eps}-{min_sample}-{reduction_type}.csv', index=False)
        #pickle.dump(dbscan, open(f'./dbscan/models/{metric}-{eps}-{min_sample}.pkl', 'wb'))

        return try_result


def parse_args():
    parser = utils.default_args('Execute DBSCAN clustering in text dataset')
    parser.add_argument('--eps-start', help='the default EPS number to start clustering', type=float, default=0.01)
    parser.add_argument('--eps-stop', help='the default EPS number to stop clustering', type=float, default=5.0)
    parser.add_argument('--eps-step', help='the default step number to increment in the EPS value', type=float,
                        default=0.001)
    parser.add_argument('--min-samples-start', help='the number of min samples to start clustering', type=int,
                        default=5)
    parser.add_argument('--min-samples-start', help='the number of max samples to stop clustering', type=int,
                        default=50)
    parser.add_argument('--min-samples-step', help='the default step number to increment in the min samples value',
                        type=int, default=1)
    parser.add_argument('--metrics', help='the distance metrics to try', type=str, default='all',
                        choices=['all', 'cosine', 'euclidean', 'manhattan'])
    parser.add_argument('--output-dir', help='the directory to save the generate artifacts', type=str,
                        default=f'{tmp.tempdir}{os.path.sep}dbscan')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args.dataset, args.stop_words, args.lem, args.stem, args.min_df, args.max_df, args.eps_start,
             args.eps_stop, args.eps_step, args.min_sample_start, args.min_sample_stop, args.min_sample_step,
             args.metrics, args.matrix, args.n_components, args.random_state, args.output_dir)
    except Exception:
        logger.error(utils.full_stack())
