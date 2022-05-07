#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from timeit import default_timer as timer
from datetime import timedelta

import preprocess
import tfidf
import utils
from loggerfactory import LoggerFactory
import pandas as pd
import numpy as np
import tempfile as tmp
from sklearn.cluster import AgglomerativeClustering


logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")
default_random_state = None


def main(dataset, stop_words, lem, stem, min_df, max_df, k_start, k_stop, k_step, linkages, affinities, matrix,
         n_components, random_state, output_dir):
    logger.info(f'Starting AgglomerativeClustering')

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
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}')

    if matrix == 'original':
        loop(df, X, k_start, k_stop, k_step, linkages, affinities, None, output_dir)
    elif matrix == 'all':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, X, k_start, k_stop, k_step, linkages, affinities, None, output_dir)
        loop(df, Y_pca, k_start, k_stop, k_step, linkages, affinities, "PCA", output_dir)
        loop(df, Y_tsne, k_start, k_stop, k_step, linkages, affinities, "TSNE", output_dir)
    elif matrix == 'pca':
        pca, Y_pca = utils.pca_reduction(X.toarray(), n_components, random_state)
        loop(df, Y_pca, k_start, k_stop, k_step, linkages, affinities, "PCA", output_dir)
    else:
        tsne, Y_tsne = utils.tsne_reduction(X, n_components, random_state)
        loop(df, Y_tsne, k_start, k_stop, k_step, linkages, affinities, "TSNE", output_dir)


def loop(df, X, k_start, k_stop, k_step, linkages, affinites, reduction_type, output_dir):
    if linkages == 'all':
        linkage = ['ward', 'complete', 'average', 'single']
    else:
        linkage = [linkages]

    if affinites == 'all':
        affinity = ['cosine', 'euclidean', 'manhattan']
    else:
        affinity = [affinites]

    results = pd.DataFrame()
    for i in np.arange(start=k_start, stop=k_stop, step=k_step):
        for link in linkage:
            for aff in affinity:
                if link == 'ward':
                    results = results.append(agglomerative(df, X, i, link, 'euclidean', reduction_type, output_dir))
                else:
                    results = results.append(agglomerative(df, X, i, link, aff, reduction_type, output_dir))

    results.to_csv(f'{output_dir}results-{reduction_type}.csv', index=False)


def agglomerative(df, X, K, link, affinity, reduction_type, output_dir):
    start = timer()

    agglo = AgglomerativeClustering(n_clusters=K, linkage=link, affinity=affinity)
    X_t = agglo.fit_predict(X.toarray() if reduction_type is None else X)
    end = timer()

    description = f'Metrics of Agglomerative with K {K}, linkage {link}, affinity {affinity} and reduction type ' \
                  f'{reduction_type} are: '

    silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski, davies, duration = \
        utils.calc_metrics(X, X_t, reduction_type, start, end)

    text = utils.log_metrics(description, silhouette_euclidian, silhouette_cosine, silhouette_manhattan, calinski,
                             davies, duration)

    logger.debug(text)
    df['cluster'] = X_t

    try_result = pd.DataFrame(data=[[K, link, affinity, reduction_type, silhouette_euclidian, silhouette_cosine,
                                     silhouette_manhattan, calinski, davies, duration]],
                              columns=['K', 'linkage', 'affinity', 'reduction_type', 'silhouette_euclidian',
                                       'silhouette_cosine', 'silhouette_manhattan', 'calinski', 'davies', 'time'])
    logger.info(f'Saving 2d and 3d plots.')
    name = str(agglo)

    utils.d2(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}2d',
             footnote=text)
    utils.d3(name, X, X_t, reduction_type is None, default_random_state, f'{output_dir}plots{os.path.sep}3d',
             footnote=text)

    df.to_csv(f'{output_dir}{K}-{link}-{affinity}-{reduction_type}.csv', index=False)
    #pickle.dump(agglomerative, open(f'./agglomerative/models/{K}-{link}-{affinity}.pkl', 'wb'))

    return try_result


def parse_args():
    parser = utils.default_args('Execute agglomerative clustering in text dataset')

    parser.add_argument('--k-start', help='the minimum number of clusters', type=int, default=2)
    parser.add_argument('--k-stop', help='the maximum number of clusters', type=int, default=30)
    parser.add_argument('--k-step', help='the default step number to increment in the K value', type=int, default=1)
    parser.add_argument('--linkages', help='the linkages to try', type=str, default='all',
                        choices=['all', 'ward', 'complete', 'average', 'single'])
    parser.add_argument('--affinities', help='the affinity metrics to try', type=str, default='all',
                        choices=['all', 'cosine', 'euclidean', 'manhattan'])
    parser.add_argument('--output-dir', help='the directory to save the generate artifacts', type=str,
                        default=f'{tmp.tempdir}{os.path.sep}agglomerative')

    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args.dataset, args.stop_words, args.lem, args.stem, args.min_df, args.max_df, args.k_start, args.k_stop,
             args.k_step, args.linkages, args.affinities, args.matrix, args.n_components, args.random_state,
             args.output_dir)
    except Exception:
        logger.error(utils.full_stack())

