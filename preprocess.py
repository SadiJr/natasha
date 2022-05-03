import re
from loggerfactory import LoggerFactory
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
#nltk.download('punkt', download_dir='./nltk_packages')
#nltk.download('stopwords', download_dir='./nltk_packages')
#nltk.download('wordnet', download_dir='./nltk_packages')
#nltk.download('omw-1.4', download_dir='./nltk_packages')
import os
data = os.path.abspath(os.curdir) + os.path.sep + 'nltk_packages'
nltk.data.path.append(data)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loggerfactory import LoggerFactory

logger = LoggerFactory.get_logger(__name__, log_level="INFO")

lemmatizer = WordNetLemmatizer()


def vectorizer(text, min_df, max_df):
    logger.debug(f'Vectorizing text {text} with min df {min_df} and max df {max_df}')
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(text)
    return X, vectorizer.get_feature_names_out()


def preprocess(x, stop_words, lematize):
    logger.debug(f'Preprocessing string {x}.')

    x = x.lower()
    x = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', x)

    if stop_words:
        tokens = nltk.word_tokenize(x)
        if lematize:
            tokens = [lemmatizer.lemmatize(w, 'v') for w in tokens if not w.lower() in stopwords.words("english")]
        else :
            tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        x = " ".join(tokens)
    x = x.strip()

    logger.debug(f'Processed string: {x}.')
    return x