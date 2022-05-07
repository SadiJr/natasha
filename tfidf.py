#!/usr/bin/env python
# coding: utf-8

from loggerfactory import LoggerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

logger = LoggerFactory.get_logger(__name__, log_level="INFO")


def tfidf(text, min_df, max_df):
    logger.info(f'Applying vectorizer TF-IDF to data with min_df {min_df} and max_df {max_df}.')
    X, terms = vectorizer(text, min_df, max_df)

    logger.debug(f'Generated sparse matrix have shape: {X.shape}.')
    logger.debug(f'And sample: {X.toarray()}')

    logger.debug(f'Total of terms: {terms.shape[0]}')
    logger.debug(f'Terms: {terms}')

    return X, terms


def vectorizer(text, min_df, max_df):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(text)
    return X, vectorizer.get_feature_names_out()
