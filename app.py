from logger import LoggerFactory
import pandas as pd
import base64
import io
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from flask import Flask
import utils
import clustering
import preprocess
import plots
import exploratory


X = None
df = pd.read_csv('data/data-processed.csv')
logger = LoggerFactory.get_logger(__name__, log_level='DEBUG')

external_stylesheets = [
    'https://fonts.googleapis.com/css?family=Open+Sans:300,400,700',
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)
app.title = 'Alfa | Agrupamento de Histórias de Usuários'

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('ufsc.svg'), style={
                'width': '80px',
                'heigth': '80px',
            }),
            html.H6('Agrupamento de Histórias de Usuários | Alfa Version', style={'font-size': '20px'}),
        ]),
    ], style={
        'text-align': 'center',
        'text-underline-position': 'under',
        'width': '100%',
        'height': '200px',
        'border-bottom': '2px solid #131313'
    }),

    html.Div([
        dcc.Upload(
                id='upload',
                children=html.Div([
                        html.A(
                            'Selecionar arquivo',
                        ),
                    ], style={
                        'text-align': 'center',
                        'text-underline-position': 'under',
                        'width': '100%',
                        'border-bottom': '2px solid #131313'
                    },
                ),
                multiple=False
            ),
    ], style={
        'font-size': '15px',
        'display': 'block',
        'align': 'center',
        'width': '100%',
        'text-decoration': 'underline'
    }),

    html.Div([
        html.Label("Operação"),
        dcc.RadioItems(
            id='operations',
            options=[
                {'label': 'Análise Exploratória', 'value': 'exp'},
                {'label': 'Agrupar todos', 'value': 'ag'},
                {'label': 'Agrupar por persona', 'value': 'per'},
                {'label': 'Apenas visualizar dados', 'value': 'vi'},
                {'label': 'Demonstração', 'value': 'demo'},
            ],
            value='demo',
            style={'display': 'grid'}
        ),
    ], style={
        'font-size': '15px',
        'display': 'block',
        'align': 'center',
        'width': '100%',
        'text-align': 'center',
        'border-bottom': '2px solid #131313'
    }),

    html.Div(
        id='exploratory-analysis',
        hidden=True,
    ),

    html.Div([
        html.Label("Algoritmo"),
        dcc.RadioItems(
            id='algorithm',
            options=[
                {'label': 'K-Means', 'value': 'kmeans'},
                {'label': 'Agglomerative', 'value': 'ag'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'GMM', 'value': 'gmm'}
            ],
            value='kmeans',
            style={'display': 'grid'}
        ),
    ], style={
        'font-size': '15px',
        'display': 'block',
        'align': 'center',
        'width': '100%',
        'text-align': 'center',
        'border-bottom': '2px solid #131313'
    }),

    html.Div([
        html.Label("Parâmetros"),
        dcc.Input(
            id='n_clusters',
            type='number',
            min=-1,
            value=-1,
            debounce=True,
        ),
    ], style={
        'font-size': '15px',
        'display': 'block',
        'align': 'center',
        'width': '100%',
        'text-align': 'center',
        'border-bottom': '2px solid #131313'
    }),

    html.Div([
        html.Label("Visualização"),
        dcc.RadioItems(
            id='plot',
            options=[
                {'label': '2D PCA', 'value': '2dpca'},
                {'label': '2D TSNE', 'value': '2dtsne'},
                {'label': '3D PCA', 'value': '3dpca'},
                {'label': '3D TSNE', 'value': '3dtsne'}
            ],
            value='2dpca',
            style={'display': 'grid'}
        ),
    ], style={
        'font-size': '15px',
        'display': 'block',
        'align': 'center',
        'width': '100%',
        'text-align': 'center',
        'border-bottom': '2px solid #131313'
    }),

    html.Div([
            html.Label('Resultados'),
        ], style={
            'font-size': '15px',
            'display': 'block',
            'align': 'center',
            'width': '100%',
            'text-align': 'center',
        }
    ),

    html.Div([
        dcc.Graph(
            id='cluster_visualization',
            style={
                'display': 'block',
                'height': '800px',
                'width': '100%'
            }
        ),
        html.Div(
            id='descriptive_statistics',
            style={
                'display': 'block',
                'height': '18vw',
                'width': '44vw'
            }
        ),
    ], style={
        'vertical-align': 'center',
        'width': '100%',
        'font-size': '12px'
    }),
])


def update_figure(value, df, X, Y=None):
    text = df.text if df is not None else None
    user = None
    if Y is not None:
        user = Y
    elif df is not None:
        user = df.user_name

    print(f'user: {user}')
    if value == '2dpca':
        return plots.pca(X, user, text)
    elif value == '2dtsne':
        return plots.tsne(X, user, text)
    elif value == '3dpca':
        return plots.pca(X, user, text, d=True)
    elif value == '3dtsne':
        return plots.tsne(X, user, text, d=True)


def operation(value, df, users, plot, algorithm, n_clusters):
    if df is not None:
        if value == 'exp':
            return [[], False, exploratory.exploratory(df)]
        elif value == 'ag':
            X = vect(df)
            Y = run_clustering(algorithm, len(df), X, n_clusters)
            print(f'Predict is {Y}')
            return [update_figure(plot, df, X, Y=Y), True, '']
        elif value == 'per':
            return [update_figure(plot, df, vect(df)), True, '']
        elif value == 'vi':
            return [update_figure(plot, None, vect(df)), True, '']
        elif value == 'demo':
            return [update_figure(plot, df, vect(df)), True, '']
        else:
            raise Exception(f'Invalid value {value}.')
    else:
        return [None, True, '']


@app.callback(
    [
        Output('cluster_visualization', 'figure'),
        Output('exploratory-analysis', 'hidden'),
        Output('exploratory-analysis', 'children')
    ],
    [
        Input('upload', 'contents'),
        Input('upload', 'filename'),
        Input('algorithm', 'value'),
        Input('plot', 'value'),
        Input('n_clusters', 'value'),
        Input('operations', 'value')
    ]
)
def update_results(uploaded_file,
                   filename,
                   algorithm,
                   plot,
                   n_clusters,
                   operation_action):
    try:
        logger.debug('Starting application...')
        if uploaded_file is not None:
            logger.debug(f'Trying to use uploaded data {uploaded_file.split(",")[0]}.')
            try:
                content_type, content_string = uploaded_file.split(',')
                decoded = base64.b64decode(content_string)

                if 'csv' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    df = pd.read_excel(io.BytesIO(decoded))
                elif 'txt' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
                else:
                    return ['', False, html.Div([html.H6('There was an error processing this file.')])]
            except Exception as e:
                logger.exception(utils.full_stack())
                return ['', False, html.Div([html.H6('There was an error processing this file.')])]
        else:
            logger.debug('Using default data.')
            df = pd.read_csv('data/data-processed.csv')

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df = utils.discard_wrong_user_stories(df)
        df, users = preprocess.label_user_types(df)

        return operation(operation_action, df, users, plot, algorithm, n_clusters)#[update_figure(plot, df, default_plot(df, users))]

    except Exception as e:
        logger.exception(utils.full_stack())
        raise dash.exceptions.PreventUpdate


def vect(df):
    df['clean'] = df.text.apply(lambda x: preprocess.decontracted(x))
    df['clean'] = df.clean.apply(lambda x: preprocess.clean(x))
    X, vec = preprocess.vectorizer(df.clean, 1, 1.0)
    return X


def run_clustering(algorithm, rows, X, k):
    if algorithm == 'kmeans':
        return clustering.kmeans(X, k, rows)
    elif algorithm == 'agglomerative':
        return clustering.agglomerative(X, k)
    elif algorithm == 'dbscan':
        return clustering.dbscan(X, 0.1, 2)
    elif algorithm == 'gmm':
        return clustering.gmm(X, k)
    else:
        return html.Div([
            'Escolha um algoritmo válido!!!'
        ])


if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
