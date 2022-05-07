import re
from loggerfactory import LoggerFactory
import nltk
# nltk.download('punkt', download_dir='./nltk_packages')
# nltk.download('stopwords', download_dir='./nltk_packages')
# nltk.download('wordnet', download_dir='./nltk_packages')
# nltk.download('omw-1.4', download_dir='./nltk_packages')
import os

data = os.path.abspath(os.curdir) + os.path.sep + 'nltk_packages'
nltk.data.path.append(data)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

logger = LoggerFactory.get_logger(__name__, log_level='INFO')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def clean_text(df, text_column='text', cleaned_column='clean', stop_words=True, lemmatize=True, stem=True):
    logger.info(f'Cleaning dataset.')
    df[cleaned_column] = df[text_column].apply(lambda x: preprocess(x, stop_words, lemmatize, stem))
    logger.info(f'Dataset clean text sample: {df.head()}')
    return df


def preprocess(text, stop_words, lemmatize, stem):
    logger.debug(f'Preprocessing string {text}.')

    text = text.lower()
    text = re.sub('[0-9]|,|\.|/|$|\(|\)|-|\+|:|•', ' ', text)

    if stop_words:
        tokens = nltk.word_tokenize(text)
        if lemmatize and stem:
            tokens = [stemmer.stem(lemmatizer.lemmatize(w, 'v')) for w in tokens if
                      not w.lower() in stopwords.words("english")]
        elif lemmatize:
            tokens = [lemmatizer.lemmatize(w, 'v') for w in tokens if not w.lower() in stopwords.words("english")]
        elif stem:
            tokens = [stemmer.stem(w) for w in tokens if not w.lower() in stopwords.words("english")]
        else:
            tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        text = " ".join(tokens)
    text = text.strip()

    logger.debug(f'Processed string: {text}.')
    return text
