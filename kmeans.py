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
from sklearn.cluster import KMeans
import tempfile as tmp
import os


default_random_state = None
logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')


def main(dataset, stop_words, lem, stem, min_df, max_df, k_start, k_stop, k_step, interaction_start, interaction_stop,
         interaction_step, n_init_start, n_init_stop, n_init_step, matrix, n_components, random_state, output_dir):
    logger.info('Starting Kmeans.')

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
        loop(df, X, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, None, output_dir)
    elif matrix == 'all':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, X, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, None, output_dir)
        loop(df, Y_pca, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, "PCA", output_dir)
        loop(df, Y_tsne, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, "TSNE", output_dir)
    elif matrix == 'pca':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        loop(df, Y_pca, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, "PCA", output_dir)
    else:
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, Y_tsne, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
             n_init_stop, n_init_step, random_state, "TSNE", output_dir)


def loop(df, X, k_start, k_stop, k_step, interaction_start, interaction_stop, interaction_step, n_init_start,
         n_init_stop, n_init_step, random_state, reduction_type, output_dir):
    results = pd.DataFrame()

    np.arange(start=k_start, stop=k_stop, step=k_step)
    for i in np.arange(start=interaction_start, stop=interaction_stop, step=interaction_step):
        for j in np.arange(start=interaction_start, stop=interaction_stop, step=interaction_step):
            for k in np.arange(start=n_init_start, stop=n_init_stop, step=n_init_step):
                results = results.append(kmeans(df, X, i, j, k, random_state, reduction_type, output_dir))

    results.to_csv(f'{output_dir}results-{reduction_type}.csv', index=False)


def kmeans(df, X, K, max_iter, n_init, random_state, reduction_type, output_dir):
    start = timer()

    km = KMeans(n_clusters=K, max_iter=max_iter, n_init=n_init, random_state=random_state)
    X_t = km.fit_predict(X)
    end = timer()

    description = f'Metrics of KMeans with K {K}, max_iter {max_iter}, n_init {n_init}, random_state {random_state} ' \
                  f'and reduction_type {reduction_type} are: '

    silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration = \
        utils.calc_metrics(X, X_t, reduction_type, start, end)

    text = utils.log_metrics(description, silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski,
                             davies, duration)
    logger.debug(text)
    df['cluster'] = X_t

    try_result = pd.DataFrame(data=[[K, max_iter, n_init, random_state, reduction_type, silhouette_euclidian,
                                     silhouette_cosine, silhouette_manhattan, calinski, davies, duration]],
                              columns=['K', 'max_iter', 'n_init', 'random_state', 'reduction_type',
                                       'silhouette_euclidian', 'silhouette_cosine', 'silhouette_manhattan', 'calinski',
                                       'davies', 'time'])
    logger.info('Saving 2d and 3d plots.')
    name = str(km)

    utils.d2(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}2d',
             footnote=text)
    utils.d3(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}3d',
             footnote=text)

    df.to_csv(f'{output_dir}{K}-{max_iter}-{n_init}-{random_state}-{reduction_type}.csv', index=False)

    #centroids = km.cluster_centers_.argsort()[:, ::-1]

    #for i in range(K):
    #    utils.word_cloud('./kmeans/figures/wordcloud', name, i, terms[centroids[i, :10]], True, footnote=text)

    df.to_csv(f'./kmeans/csvs/{K}-{max_iter}-{n_init}-{random_state}-{reduction_type}.csv')
    #pickle.dump(km, open(f'./kmeans/models/{K}-{max_iter}-{n_init}.pkl', 'wb'))

    return try_result


def parse_args():
    parser = utils.default_args('Execute KMeans clustering in text dataset')
    parser.add_argument('--k-start', help='the default EPS number to start clustering', type=float, default=0.01)
    parser.add_argument('--k-stop', help='the default EPS number to stop clustering', type=float, default=5.0)
    parser.add_argument('--k-step', help='the default step number to increment in the EPS value', type=float,
                        default=0.001)
    parser.add_argument('--interaction-start', help='the number of min samples to start clustering', type=int,
                        default=5)
    parser.add_argument('--interaction-stop', help='the number of max samples to stop clustering', type=int,
                        default=50)
    parser.add_argument('--interaction-step', help='the default step number to increment in the min samples value',
                        type=int, default=1)
    parser.add_argument('--n-init-start', help='the number of min samples to start clustering', type=int,
                        default=5)
    parser.add_argument('--n-init-stop', help='the number of max samples to stop clustering', type=int,
                        default=50)
    parser.add_argument('--n-init-step', help='the default step number to increment in the min samples value',
                        type=int, default=1)
    parser.add_argument('--output-dir', help='the directory to save the generate artifacts', type=str,
                        default=f'{tmp.tempdir}{os.path.sep}kmeans')

    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args.dataset, args.stop_words, args.lem, args.stem, args.min_df, args.max_df, args.k_start, args.k_stop,
             args.k_step, args.interaction_start, args.interaction_stop, args.interaction_step, args.n_init_start,
             args.n_init_stop, args.n_init_step, args.matrix, args.n_components, args.random_state, args.output_dir)
    except Exception:
        logger.error(utils.full_stack())
