import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from model.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',
              1000, 'max_colwidth', 1000)

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def clean(x):
    x = x.lower()
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
    x = re.sub('\s\s+', ' ', x)
    return x.strip()


# https://stackoverflow.com/a/47091490
def decontracted(phrase):
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


def count_contractions(texts):
    return texts.str.count("'").sum()


def remove_double_spaces(x):
    x = re.sub('\s\s+', ' ', x)
    return x.strip()


def find_user_entity(x):
    try:
        entity = re.match("^As (an|a)[\s\w/-]*,\s*I\s*", x)
        name = re.sub('^As\s*(an|a)\s*', '', entity.group(0))
        return name.split(',')[0].strip().lower()
    except Exception as e:
        return None


def set_user_cluster(all_users, user):
    user_type = find_user_entity(user)
    if user_type is None or user_type not in all_users:
        return -1
    return all_users.index(user_type)


def label_user_types(df):
    all_types = list(filter(None, df.text.apply(lambda x: find_user_entity(x)).tolist()))
    users = list(dict.fromkeys(all_types))
    df['user_type'] = df.text.apply(lambda x: set_user_cluster(users, x))
    df.drop(df.loc[df.user_type == -1].index, inplace=True)
    return df, users


def check_user_story_pattern(x):
    matchs = re.match("^As (an|a) [\w\W]*, I [\w\W]*, so [\w\W]+\.$", x)
    return matchs is not None


def discard_wrong_user_stories(df):
    df['correctly'] = df['clean'].apply(lambda x: check_user_story_pattern(x))
    to_process = df.loc[df.correctly]
    to_process.reset_index(inplace=True)
    return to_process


def preprocess(x, stop_words, lem, stem):
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
    x = re.sub("\'", '', x)
    x = re.sub('\s\s+', ' ', x)
    x = x.strip()

    logger.debug(f'Processing user story: [{x}]')

    stops = stopwords.words('english')
    stops.append('want')
    tokens = nltk.word_tokenize(x)

    if stop_words:
        tokens = [w for w in tokens if not w.lower() in stops]

    if lem:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    if stem:
        tokens = [stemmer.stem(w) for w in tokens]

    x = ' '.join(filter(lambda token: len(token) > 2, tokens))
    x = x.strip()
    x = re.sub('\s\s+', ' ', x)

    logger.debug(f'Processed user story: [{x}]')
    return x


def vectorizer(text, min_df, max_df):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(text)
    return X, vectorizer


def read_user_stories(path):
    if not os.path.isfile(path):
        err = f'Cannot find file with user stories to process in path: [{path}]'
        logger.error(err)
        raise Exception(err)
    tmp_df = pd.read_csv(path)
    tmp_df['clean'] = tmp_df.text.apply(lambda x: remove_double_spaces(x))
    tmp_df['clean'] = tmp_df['clean'].apply(lambda x: decontracted(x))
    tmp_df, users = label_user_types(tmp_df)

    df = discard_wrong_user_stories(tmp_df)
    df['clean'] = df['clean'].apply(lambda x: x.lower())
    df['clean'].drop_duplicates(inplace=True)

    return df
