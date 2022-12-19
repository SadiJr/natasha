import re
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from model.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

_path = os.path.dirname(__file__)


class TextAction:
    _default_options = {}

    def apply(self, df):
        pass

    def clean(self, x):
        #logger.debug(f'Cleaning text: {x}')
        x = x.lower()
        x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•|\_', ' ', x)
        x = re.sub("\'", '', x)
        x = re.sub('\s\s+', ' ', x)
        #logger.debug(f'Cleaned text: {x.strip()}')
        return x.strip()

    # https://stackoverflow.com/a/47091490
    def decontracted(self, phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"facilities\'", "facilities", phrase)
        phrase = re.sub(r"users\'", "users", phrase)
        phrase = re.sub(r"sites\'", "sites", phrase)
        phrase = re.sub(r"campers\'", "campers", phrase)
        phrase = re.sub(r"Nodes\'", "Nodes", phrase)
        phrase = re.sub(r"Archivists\'", "Archivists", phrase)
        phrase = re.sub(r"researchers\'", "researchers", phrase)
        phrase = re.sub(r"others\'", "others", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def preprocess(self, x):
        #logger.debug(f'Preprocessing text: {x}')
        x = self.decontracted(x)
        x = self.clean(x)

        tokens = nltk.word_tokenize(x)
        x = ' '.join(filter(lambda token: len(token) > 2, tokens))
        x = x.strip()
        x = re.sub('\s\s+', ' ', x)

        #logger.debug(f'Preprocessed text: {x}')
        return x

    @classmethod
    def get_options(cls):
        return cls._default_options


class Default(TextAction):
    def apply(self, df):
        preprocessed = [self.preprocess(text) for text in df.values.ravel()]
        return pd.DataFrame(preprocessed, columns=df.columns)


class Stem(TextAction):
    def apply(self, df):
        preprocessed = [self.preprocess(text) for text in df.values.ravel()]

        processed = []
        for phrase in preprocessed:
            #logger.debug(f'Processing phrase: {phrase}')
            tokens = nltk.word_tokenize(phrase)
            tokens = [SnowballStemmer('english').stem(w) for w in tokens]

            x = ' '.join(filter(lambda token: len(token) > 2, tokens))
            x = x.strip()
            x = re.sub('\s\s+', ' ', x)
            processed.append(x)
            #logger.debug(f'Processed phrase: {x}')

        logger.debug(f'Processed dataframe is: {processed}')
        return pd.DataFrame(processed, columns=df.columns)


class Lemmatize(TextAction):
    def apply(self, df):
        lemm = WordNetLemmatizer()

        preprocessed = [self.preprocess(text) for text in df.values.ravel()]
        #preprocessed = list(dict.fromkeys(preprocessed))
        processed = []
        for phrase in preprocessed:
            #logger.debug(f'Processing phrase: {phrase}')
            tokens = nltk.word_tokenize(phrase)
            tokens = [lemm.lemmatize(w) for w in tokens]

            x = ' '.join(filter(lambda token: len(token) > 2, tokens))
            x = x.strip()
            x = re.sub('\s\s+', ' ', x)
            processed.append(x)
            #logger.debug(f'Processed phrase: {x}')

        logger.debug(f'Processed dataframe is: {processed}')
        return pd.DataFrame(processed, columns=df.columns)


class StopWords(TextAction):
    def apply(self, df):
        preprocessed = [self.preprocess(text) for text in df.values.ravel()]

        stops = stopwords.words('english')
        stops.append('want')

        processed = []
        for phrase in preprocessed:
            #logger.debug(f'Processing phrase: {phrase}')
            tokens = nltk.word_tokenize(phrase)
            tokens = [w for w in tokens if not w.lower() in stops]
            x = ' '.join(filter(lambda token: len(token) > 2, tokens))
            x = x.strip()
            x = re.sub('\s\s+', ' ', x)
            processed.append(x)
            #logger.debug(f'Processed phrase: {x}')

        logger.debug(f'Processed dataframe is: {processed}')
        return pd.DataFrame(processed, columns=df.columns)


class TFIDF(TextAction):
    _default_options = {
        "max_df": 1.0, "min_df": 1
    }

    def __init__(self, max_df=1.0, min_df=1):
        self.options = {"max_df": float(max_df), "min_df": float(min_df) if min_df.count('.') > 0 else int(min_df)}

    def get_vec_arr(self, df):
        logger.debug(f'Applying TF-IDF in dataframe using options {self.options}')
        countvec = TfidfVectorizer(**{**self.options})
        countarr = countvec.fit_transform(df['text_to_cluster'])

        logger.debug(f'Generated TF-IDF matrix is:\n{pd.DataFrame(countarr.toarray(), columns=countvec.get_feature_names_out())}')
        return countvec, countarr

    def apply(self, df):
        vec, arr = self.get_vec_arr(df)
        return pd.DataFrame(arr.toarray(), columns=vec.get_feature_names_out())


def join_columns(df, chosen_cols):
    logger.debug(f'chosen_cols: {chosen_cols}')
    series = df[chosen_cols[0]].astype(str)

    #logger.debug(f'series: {series}')
    for col in chosen_cols[1:]:
        series = series + ". " + df[col].astype(str)
        logger.debug(f'series: {series}')

    #logger.debug(f'series.values: {series.values}')
    df = pd.DataFrame(series.values, columns=["text_to_cluster"])

    return df
