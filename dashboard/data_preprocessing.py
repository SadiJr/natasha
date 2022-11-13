from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Output

import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache

from model.logger import LoggerFactory
from model.utils import full_stack

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

data_preprocessing_options = {
    "Default": text_processing.Default,
    "Remover Stop Words": text_processing.StopWords,
    "Stemmatização": text_processing.Stem,
    "Lemmatização": text_processing.Lemmatize,
}

data_preprocessing_tab = dcc.Tab(
    label="Pré Processamento", children=[
        html.Div(id="text_preprocess_area", children=[
            html.Div([
                html.H5("Opções de Pré-Processamento:"),
                dcc.Checklist(id="text_preprocess_checklist",
                              options=[{"label": name, "value": name} for name, cls in data_preprocessing_options.items()], value=[],
                              style={"padding": "5px", "margin": "5px"})
            ], id="text_preprocess_checklist_div"),
            html.H5("Exemplo de texto pré-processado:"),
            html.Div(dash_table.DataTable(id="text_preprocess"), id="text_preprocess_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "preprocessing_method": misc.HtmlElement("text_preprocess_checklist", "value"),
    "preprocessing_output": misc.HtmlElement("text_preprocess_div", "children"),
}
outputs = Output("text_preprocess_div", "children")


@cache.memoize()
def get_preprocessed_data(df, preprocessing_method):
    logger.debug(f'preprocessing_method: {preprocessing_method}.')
    if df is not None and preprocessing_method:
        for method in preprocessing_method:
            df = data_preprocessing_options[method]().apply(df)

    #logger.debug(f'Generated preprocessed dataframe:\n{df}')
    return df


def get_data_preprocessing_output(df, preprocessing_method):
    df = get_preprocessed_data(df, preprocessing_method)

    return misc.generate_datatable(df, "text_preprocess", 5, max_cell_width=None)
