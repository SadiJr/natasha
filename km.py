#!/usr/bin/env python
# coding: utf-8

from timeit import default_timer as timer
from datetime import timedelta
from loggerfactory import LoggerFactory
import pandas as pd
import numpy as np
import re
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score, v_measure_score
nltk.download('punkt')
nltk.download('stopwords')


# In[ ]:

def main():

    df = read_data()

    start = timer()
    df = clean_text(df)
    end = timer()
    logger.info(f'Time to clean text: {timedelta(seconds=end - start)}')

    start = timer()
    X, terms = tfidf(df)
    end = timer()
    logger.info(f'Time to vectorizer text: {timedelta(seconds=end - start)}')

    km_loop(df, X, terms)

def read_data():
    logger.info(f'Reading dataset...')

    df = pd.read_csv('./data/concat.txt', header=None)
    df.rename(columns = {0:'text'}, inplace = True)
    df.drop(1, axis=1, inplace=True)

    logger.info(f'Dataset have {df.shape[0]} rows.')
    logger.info(f'Dataset info: {df.info()}')
    logger.info(f'Dataset sample: {df.head()}')

    return df


def clean_text(df):
    logger.info(f'Cleaning dataset.')
    df['clean'] = df.text.apply(lambda x: preprocess(x, True))
    logger.info(f'Dataset clean text sample: {df.head()}')
    return df


def tfidf(df):
    logger.info(f'Applying vectorizer TF-IDF to data')
    X, terms = vectorizer(df.clean, 5, 0.95)

    logger.info(f'Generated sparse matrix have shape: {X.shape}.')
    logger.info(f'And sample: {X.toarray()}')

    logger.info(f'Total of terms: {terms.shape[0]}')
    logger.info(f'Terms: {terms}')

    return X, terms



def vectorizer(text, min_df, max_df):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(text)
    return X, vectorizer.get_feature_names_out()


def km_loop(df, X, terms):
    for i in range (10, 50):
        for j in range (100, 500):
            for k in range (10, 100):
                for l in range (10, 50):
                    kmeans(df, X.toarray(), terms, i, 'k-means++', j, k, 'auto', l)
                    kmeans(df, X.toarray(), terms, i, 'k-means++', j, k, 'full', l)
                    kmeans(df, X.toarray(), terms, i, 'k-means++', j, k, 'elkan', l)
                    kmeans(df, X.toarray(), terms, i, 'random', j, k, 'auto', l)
                    kmeans(df, X.toarray(), terms, i, 'random', j, k, 'full', l)
                    kmeans(df, X.toarray(), terms, i, 'random', j, k, 'elkan', l)

def kmeans(df, X, terms, K, init, max_iter, n_init, algorithm, random_state):
    start = timer()

    km = KMeans(n_clusters=K, 
                init=init,
                max_iter=max_iter,
                n_init=n_init,
                algorithm=algorithm,
                random_state=random_state)
    km.fit(X)
    end = timer()

    logger.debug(f'Metrics of KMeans with {K}, init {init}, max_iter {max_iter}, n_init {n_init}, algorithm {algorithm}'
                 f' and random state {random_state}  are: ')
    logger.debug(f'Calinski-Harabasz Index: {metrics.calinski_harabasz_score(X, km.labels_)}')
    logger.debug(f'Davies-Bouldin Index: {metrics.davies_bouldin_score(X, km.labels_)}')
    logger.debug(f'Silhouette Coefficient: {metrics.silhouette_score(X, km.labels_, metric="euclidean")}')
    logger.debug(f'Total execution time: {timedelta(seconds=end - start)}')

    df['cluster'] = km.labels_
    centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(K):
        logger.debug(f'Cluster {i} top ten terms:')
        for j in centroids[i, :10]:
            logger.debug(f'{terms[j]}')
        logger.debug('----------------------------------------')

    df.to_csv(f'./output/{K}-{init}-{max_iter}-{n_init}-{algorithm}-{random_state}.csv')


def preprocess(x, stop_words):
    logger.info(f'Converting string to lower case, removing pontuation, tokenize and finally remove stop words.')

    logger.info(f'Original string: {x}')
    x = x.lower()
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
      
    if stop_words:
        tokens = nltk.word_tokenize(x)
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        x = " ".join(tokens)
    x = x.strip()

    logger.info(f'Cleaned string: {x}')
    return x


# In[ ]:


def porterStemmer(df):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    df['porter'] = df.text.apply(lambda x: stemmer.stem(x))
    return df


def snowballStemmer(df, lang):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer(lang)
    df['snowball'] = df.text.apply(lambda x: stemmer.stem(x))
    return df


def lancasterStemmer(df):
    from nltk.stem import LancasterStemmer
    stemmer = LancasterStemmer()
    df['lancaster'] = df.text.apply(lambda x: stemmer.stem(x))
    return df


if __name__ == '__main__':
    logger = LoggerFactory.get_logger(__name__, log_level="DEBUG")

    logger.info(f'Starting Kmeans')
    main()

