import inspect
import sys
import os
import dash

sys.path.append(os.getcwd())

import data.text_processing as text_processing

import dashboard.data_selection as data_selection
import dashboard.data_preprocessing as data_preprocessing
import dashboard.data_exploratory_analysis as data_analysis
import dashboard.data_to_array as data_to_array
import dashboard.data_dim_reduction as data_dim_reduction
import dashboard.plotting as plotting
import dashboard.clustering as clustering

from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dashboard.cache import cache
from model.logger import LoggerFactory
from model.utils import full_stack

logger = LoggerFactory.get_logger(__name__, 'DEBUG')
CACHE_CONFIG = {
    "CACHE_TYPE": "simple"
}

app = dash.Dash(__name__, update_title="Atualizando! Por favor, aguarde.")
app.title = 'Protótipo | Agrupamento de Histórias de Usuários'

cache.init_app(app.server, config=CACHE_CONFIG)

name_to_html_element_pool = {
    **data_selection.arguments,
    **data_preprocessing.arguments,
    **data_to_array.arguments,
    **data_dim_reduction.arguments,
    **plotting.arguments,
    **clustering.arguments
}

processed = None


def map_arguments(outputs):
    logger.debug(f'Mapping outputs: {outputs}.')
    def _map_arguments(func):
        inputs = []
        states = []
        for argument in inspect.getfullargspec(func).args:
            if argument.startswith("s_"):
                states.append(State(*name_to_html_element_pool[argument.replace("s_", "")]))
            else:
                inputs.append(Input(*name_to_html_element_pool[argument]))

        return app.callback(outputs, inputs, states)(func)

    logger.debug(f'Generated maps: {_map_arguments}')
    return _map_arguments


################################################################################################
app.layout = html.Div([
    html.Div([
        html.Img(src=app.get_asset_url('ufsc.svg'), style={'height': '100px', 'display': 'inline-block',
                                                           'vertical-align': 'bottom'}),
        html.H1("Agrupamento de Histórias de Usuários", style={'display': 'inline-block', 'vertical-align': 'middle',
                                                               'padding': '0px 50px'}),
        html.P(children=None, style={"padding": "5px", "margin": "5px"})
    ], style={'text-align': 'center'}),
    html.Div([
        dcc.Tabs(id="tabs_1", children=[
            data_selection.data_selection_tab,
            data_analysis.data_exploratory_tab,
            data_preprocessing.data_preprocessing_tab,
            data_to_array.data_to_array_tab,
            data_dim_reduction.dim_reduction_tab,
        ]),
    ], style={"border": "grey solid", "padding": "5px"}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            plotting.plotting_options_tab,
            clustering.clustering_tab,
        ]),
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"}),

    html.Div(id="plot_area", children=[
        dcc.Tabs(id="tabs_3", children=[
            plotting.plot_tab,
            clustering.clusters_tab
        ], style={"border": "grey solid", "padding": "5px", "marginTop": "10px"})
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"}),

    html.Div([
        html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
    ]),
], style={"background-color": "#f2f2f2", "margin": "20px"})


################################################################################################
def get_data(selected_data):
    logger.debug(f'Trying to get data: {selected_data}.')
    return data_selection.get_data(selected_data)


def get_data_selected_columns(selected_data, selected_columns):
    logger.debug(f'Trying to get column [{selected_columns}] of selected data [{selected_data}].')
    df = get_data(selected_data)
    return data_selection.get_selected_columns(df, selected_columns)


def get_data_preprocessed(selected_data, selected_columns, preprocessing_method):
    logger.debug(f'Trying to get column [{selected_columns}] of selected data [{selected_data}] '
                 f'using preprocessing method: [{preprocessing_method}].')
    df = get_data_selected_columns(selected_data, selected_columns)
    return data_preprocessing.get_preprocessed_data(df, preprocessing_method)


def get_data_as_array(selected_data, selected_columns, preprocessing_method,
                      data_to_array_method, data_to_array_options):
    logger.debug(f'Trying to generate vector data of column [{selected_columns}] of selected data [{selected_data}] '
                 f'using preprocessing method: [{preprocessing_method}], vectorizer method [{data_to_array_method}]'
                 f'and vectorizer options: [{data_to_array_options}].')
    df = get_data_preprocessed(selected_data, selected_columns, preprocessing_method)
    df_arr = data_to_array.get_data_as_array(df, data_to_array_method, data_to_array_options)

    logger.debug(f'Readed dataframe: [{df}] and vector: [{df_arr}].')
    return df, df_arr


def get_data_as_array_dim_red(selected_data, selected_columns,
                              preprocessing_method,
                              data_to_array_method, data_to_array_options,
                              dim_reduction, dim_reduction_options):
    logger.debug(f'Trying to generate reduced vector data of column [{selected_columns}] of selected data [{selected_data}] '
                 f'using preprocessing method: [{preprocessing_method}], vectorizer method [{data_to_array_method}], '
                 f'vectorizer options: [{data_to_array_options}] and dimension reduction [{dim_reduction}] with '
                 f'options: [{dim_reduction_options}].')
    df, df_arr = get_data_as_array(selected_data, selected_columns,
                                   preprocessing_method,
                                   data_to_array_method, data_to_array_options)
    df_arr_dim_red = df_arr
    if dim_reduction and dim_reduction_options:
        df_arr_dim_red = data_dim_reduction.get_dim_reduction(df_arr, dim_reduction, dim_reduction_options)

    logger.debug(f'Readed dataframe: [{df}]\nVector generated: [{df_arr}]\nReduced vector: [{df_arr_dim_red}]')
    return df, df_arr, df_arr_dim_red


def get_data_clustered(selected_data, selected_columns,
                       preprocessing_method,
                       data_to_array_method, data_to_array_options,
                       dim_reduction_method, dim_reduction_options,
                       clustering_method, clustering_options):
    logger.debug(
        f'Trying to generate clusters with parameters:\nselected_data: {selected_data}\nselected_columns: '
        f'{selected_columns}\npreprocessing_method: {preprocessing_method}\ndata_to_array_method: '
        f'{data_to_array_method}\ndata_to_array_options: {data_to_array_options}\ndim_reduction_method: '
        f'{dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}\nclustering_method: '
        f'{clustering_method}\nclustering_options: {clustering_options}')
    _, df_arr, df_arr_dim_red = get_data_as_array_dim_red(
        selected_data, selected_columns,
        preprocessing_method, data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options
    )

    clusters = clustering.get_clusters(get_data(selected_data), df_arr_dim_red, clustering_method, clustering_options)

    logger.debug(f'Generated data are:\n_: {_}\ndf_arr: {df_arr}\ndf_arr_dim_red: {df_arr_dim_red}\nclusters: {clusters}')
    return _, df_arr, df_arr_dim_red, clusters


################################################################################################
data_to_array.data_to_array_dropdown.generate_update_options_callback(app)
data_dim_reduction.dim_reduction_dropdown.generate_update_options_callback(app)
plotting.plotting_options_dropdown.generate_update_options_callback(app)
clustering.clustering_dropdown.generate_update_options_callback(app)


################################################################################################
@map_arguments(data_selection.outputs)
def update_data_selection(
        selected_data
):
    logger.debug(f'Updating data selection with input: {selected_data}.')
    return data_selection.get_data_selection_output(selected_data)


@map_arguments(data_preprocessing.outputs)
def update_data_preprocessing(
        selected_columns, preprocessing_method,
        s_selected_data
):
    logger.debug(f'selected_columns: {selected_columns}\npreprocessing_method: {preprocessing_method}\n'
                 f's_selected_data: {s_selected_data}.')
    df = get_data_selected_columns(s_selected_data, selected_columns)

    return data_preprocessing.get_data_preprocessing_output(df, preprocessing_method)


@map_arguments(data_analysis.outputs)
def update_exploratory_analysis(
    selected_data, selected_columns
):
    df = get_data(selected_data)
    return data_analysis.exploratory(df, selected_columns)


@map_arguments(data_to_array.outputs)
def update_data_to_array(
        data_to_array_method,
        data_to_array_options,

        data_to_array_refresh,
        preprocessing_output,

        s_selected_data, s_selected_columns, s_preprocessing_method,
):
    logger.debug(f'data_to_array_method: {data_to_array_method}\ndata_to_array_options: {data_to_array_options}\n'
                 f'data_to_array_refresh: {data_to_array_refresh}\npreprocessing_output: {preprocessing_output}\n'
                 f's_selected_data: {s_selected_data}\ns_selected_columns: {s_selected_columns}\n'
                 f's_preprocessing_method: {s_preprocessing_method}.')
    df = get_data_preprocessed(s_selected_data, s_selected_columns, s_preprocessing_method)
    return data_to_array.get_data_to_array_output(df, data_to_array_method, data_to_array_options)


@map_arguments(data_dim_reduction.outputs)
def update_dim_reduction(
        dim_reduction_method,
        dim_reduction_options,

        dim_reduction_refresh,
        data_to_array_table,

        s_selected_data, s_selected_columns, s_preprocessing_method, s_data_to_array_method, s_data_to_array_options,
):
    logger.debug(f'dim_reduction_method: {dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}\n'
                 f'dim_reduction_refresh: {dim_reduction_refresh}\ndata_to_array_table: {data_to_array_table}\n'
                 f's_selected_data: {s_selected_data}\ns_selected_columns: {s_selected_columns}\ns_preprocessing_method: '
                 f'{s_preprocessing_method}\ns_data_to_array_method: {s_data_to_array_method}\ns_data_to_array_options: '
                 f'{s_data_to_array_options}.')
    df, df_arr = get_data_as_array(s_selected_data, s_selected_columns, s_preprocessing_method, s_data_to_array_method,
                                   s_data_to_array_options)

    return data_dim_reduction.get_dim_reduction_output(df_arr, dim_reduction_method, dim_reduction_options)


@map_arguments(plotting.outputs)
def update_plot(
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        plot_dim_reduction_method, plot_dim_reduction_options,
        clustering_method, clustering_options,

        dim_reduction_refresh,
        plot_dim_reduction_refresh, clustering_refresh,

        s_selected_data, s_selected_columns, s_preprocessing_method,
):
    logger.debug(f'data_to_array_method: {data_to_array_method}\ndata_to_array_options: {data_to_array_options}\n'
                 f'dim_reduction_method: {dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}\n'
                 f'plot_dim_reduction_method: {plot_dim_reduction_method}\nplot_dim_reduction_options: '
                 f'{plot_dim_reduction_options}\nclustering_method: {clustering_method}\nclustering_options: '
                 f'{clustering_options}\ndim_reduction_refresh: {dim_reduction_refresh}\nplot_dim_reduction_refresh: '
                 f'{plot_dim_reduction_refresh}\nclustering_refresh: {clustering_refresh}\ns_selected_data: '
                 f'{s_selected_data}\ns_selected_columns: {s_selected_columns}\ns_preprocessing_method: '
                 f'{s_preprocessing_method}.')
    _, df_arr, _, clusters = get_data_clustered(
        s_selected_data, s_selected_columns, s_preprocessing_method, data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options, clustering_method, clustering_options
    )

    titles = None
    if s_selected_data:
        titles = get_data(s_selected_data).text
        logger.debug(f'Using titles: {titles}.')

    return plotting.get_plot_output(df_arr, plot_dim_reduction_method, plot_dim_reduction_options,
                                    clusters, titles)


@map_arguments(clustering.outputs)
def update_cluster_clustering(
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        plot_dim_reduction_method, plot_dim_reduction_options,
        clustering_method, clustering_options,

        dim_reduction_refresh,
        plot_dim_reduction_refresh, clustering_refresh,

        s_selected_data, s_selected_columns, s_preprocessing_method,
):
    logger.debug(f'data_to_array_method: {data_to_array_method}\ndata_to_array_options: {data_to_array_options}\n'
                 f'dim_reduction_method: {dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}\n'
                 f'plot_dim_reduction_method: {plot_dim_reduction_method}\nplot_dim_reduction_options: '
                 f'{plot_dim_reduction_options}\nclustering_method: {clustering_method}\nclustering_options: '
                 f'{clustering_options}\ndim_reduction_refresh: {dim_reduction_refresh}\nplot_dim_reduction_refresh: '
                 f'{plot_dim_reduction_refresh}\nclustering_refresh: {clustering_refresh}\ns_selected_data: '
                 f'{s_selected_data}\ns_selected_columns: {s_selected_columns}\ns_preprocessing_method: '
                 f'{s_preprocessing_method}.')
    _, _, df_arr_dim_red = get_data_as_array_dim_red(
        s_selected_data, s_selected_columns, s_preprocessing_method,
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options
    )

    titles = None
    if s_selected_data:
        titles = get_data(s_selected_data).text
        logger.debug(f'Generated titles: {titles}')

    bow = None
    if s_selected_data:
        bow = text_processing.BOW(ngram_range=(1, 1), min_df=1).apply(
            get_data_preprocessed(s_selected_data, s_selected_columns, s_preprocessing_method)
        )
        #logger.debug(f'Generated bow: {bow}.')

    return clustering.get_clustering_cluster_output(get_data(s_selected_data), df_arr_dim_red, clustering_method, clustering_options, titles, bow)


### temporary callbacks
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
def func(n_clicks):
    global df
    data = df[['text', 'user_name']]
    return dcc.send_data_frame(data.to_csv, "processed.csv")


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8080, debug=True)
