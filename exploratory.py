from dash import dcc
from dash import html
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import preprocess
import pandas as pd
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


def get_top_n_words(corpus, remove_stop=False, n=None, grams=(1, 1), remove_want=False):
    if remove_stop:
        stops = stopwords.words('english')
        if remove_want:
            stops.append('want')
        vec = CountVectorizer(stop_words=stops, ngram_range=grams).fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=grams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_distribution(data, xTitle, yTitle, title):
    return data.iplot(
        kind='hist',
        bins=100,
        xTitle=xTitle,
        linecolor='black',
        yTitle=yTitle,
        title=title,
        asFigure=True
    )


def plot_n_gramns(data, column, column2, yTitle, title):
    return data.groupby(column).sum()[column2].sort_values(ascending=False).iplot(
        kind='bar',
        yTitle=yTitle,
        linecolor='black',
        title=title,
        asFigure=True
    )


def exploratory(data):
    df = data.copy()

    df['clean'] = df.text.apply(lambda x: preprocess.decontracted(x))
    df['clean'] = df.clean.apply(lambda x: preprocess.clean(x))

    df['stories_length'] = df['clean'].astype(str).apply(len)
    df['word_count'] = df['clean'].apply(lambda x: len(str(x).split()))

    stories_distribution = plot_distribution(df['stories_length'], 'Stories length', 'Count', 'Stories Text Length Distribution')
    word_distribution = plot_distribution(df['word_count'], 'Word count', 'Count', 'Stories Text Word Count Distribution')

    common_words = get_top_n_words(df['text'], False, 20)
    df1 = pd.DataFrame(common_words, columns=['UserStory', 'count'])
    ngrams = plot_n_gramns(df1, 'UserStory', 'count', 'Count', 'Top 20 words in user stories before removing stop words')

    common_words = get_top_n_words(df['text'], True, 20)
    df1 = pd.DataFrame(common_words, columns=['UserStory', 'count'])
    another_ngrams = plot_n_gramns(df1, 'UserStory', 'count', 'Count', 'Top 20 words in stories after removing stop words')

    return html.Div([
        html.Label('Análise Exploratória'),
        html.Div([
            dcc.Graph(
                id='graph1',
                figure=stories_distribution
            ),  
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='graph2',
                figure=word_distribution
            ),
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='graph3',
                figure=ngrams
            ),
        ], className='four columns'),
        html.Div([
            dcc.Graph(
                id='graph4',
                figure=another_ngrams
            ),
        ], className='four columns'),
    ], className='row',
        style={
            'font-size': '15px',
            'display': 'block',
            'align': 'center',
            'width': '100%',
            'text-align': 'center',
            'border-bottom': '2px solid #131313'
    })
