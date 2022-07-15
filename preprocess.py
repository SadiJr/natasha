import logging
import re
from logger import LoggerFactory
import nltk
import os
import pandas as pd
from chardet import detect
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
import utils

nltk.download('punkt', download_dir='./nltk_packages')
nltk.download('stopwords', download_dir='./nltk_packages')
nltk.download('wordnet', download_dir='./nltk_packages')
nltk.download('omw-1.4', download_dir='./nltk_packages')

data = os.path.abspath(os.curdir) + os.path.sep + 'nltk_packages'
nltk.data.path.append(data)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

logger = LoggerFactory.get_logger(__name__, log_level='INFO')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def get_encoding_type(file):
    encoding = None
    with open(file, 'rb') as f:
        rawdata = f.read()
        encoding = detect(rawdata)['encoding']
    return encoding


def convert_encoding(source, codec):
    try:
        target = f'{source}-converted'
        with open(source, 'r', encoding=codec) as s, open(target, 'w', encoding='utf-8') as t:
            text = s.read()
            t.write(text)
        os.remove(source)
        os.rename(target, source)
    except UnicodeDecodeError:
        logging.exception(f'Decode error: {utils.full_stack()}')
    except UnicodeEncodeError:
        logging.exception(f'Encode error: {utils.full_stack()}')


def convert_to_utf8(directory):
    for f in os.listdir(directory):
        full_path = os.path.abspath(os.path.join(directory, f))
        encoding = get_encoding_type(full_path)
        if encoding is not None:
            convert_encoding(full_path, encoding)


def concat_all_files(directory):
    tmp = tempfile.mkstemp()[1]
    for files in os.listdir(directory):
        full_path = os.path.abspath(os.path.join(directory, files))
        encoding = get_encoding_type(full_path)
        if encoding is not None:
            with open(full_path, 'r') as f, open(tmp, 'a') as t:
                t.write(f.read())
    return tmp


def check_user_story_pattern(x):
    matchs = re.match("As (an|a)[\s\w/-]*,\s*I [\s\w]*,\sso[\s\w/-]*.", x)
    return matchs is not None


def discard_wrong_user_stories(df):
    df['correctly'] = df.text.apply(lambda x: check_user_story_pattern(x))
    to_process = df.loc[df.correctly]
    to_process.reset_index(inplace=True)
    return to_process


def read_user_stories(directory):
    convert_to_utf8(directory)
    tmp = concat_all_files(directory)
    df = pd.read_fwf(tmp, header=None, delimiter='\n', names=['text'])
    stories = discard_wrong_user_stories(df)
    return stories


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def preprocess(x, to_lower=True, special=True, stop_words=True, stem=True, lem=False):
    if to_lower:
        x = x.lower()
    if special:
        x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
        x = re.sub("\'", '', x)

    stops = stopwords.words('english')
    stops.append('want')
    tokens = nltk.word_tokenize(x)

    if stop_words:
        if lem and stem:
            tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens if not w.lower() in stops]
        elif lem:
            tokens = [lemmatizer.lemmatize(w) for w in tokens if not w.lower() in stops]
        elif stem:
            tokens = [stemmer.stem(w) for w in tokens if not w.lower() in stops]
        else:
            tokens = [w for w in tokens if not w.lower() in stops]
    elif stem:
        tokens = [stemmer.stem(w) for w in tokens]
    elif lem:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    x = ' '.join(tokens)
    x = x.strip()
    return x


def vectorizer(text, min_df, max_df):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(text)
    return X, vectorizer


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
    df['user_name'] = df.user_type.apply(lambda x: users[x] if x != -1 else None)
    df.drop(df.loc[df.user_type == -1].index, inplace=True)
    return df, users


def clean(x):
    x = x.lower()
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)
    x = re.sub('\s\s+', ' ', x)
    return x.strip()
