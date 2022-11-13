from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Output

from data.demo import get_demo_data
import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache

from model.logger import LoggerFactory
from model.utils import full_stack

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

data_sources = {
    "Histórias de Demonstração": get_demo_data
}

data_selection_tab = dcc.Tab(
    label="Dados", children=[
        html.Div(id="data_selection_area", children=[
            html.H5(children="Arquivo: "),

            dcc.Dropdown(
                id="data",
                options=[{"label": name, "value": name} for name, func in data_sources.items()],
                placeholder="Selecione uma das opções disponíveis ou ... "
            ),

            dcc.Upload(id='upload_data', children=[html.Button('Fazer upload de Arquivo')]),
            dcc.Checklist(id='column_names', options=[], labelStyle={'display': 'block'}),

            html.H5("Coluna com as Histórias de Usuários:"),
            html.Div(dcc.Dropdown(id="data_column_selector",
                                  multi=False),
                     id="data_column_selector_div"),
            html.H5("Informações sobre o Dataset:", id="data_top_rows"),
            html.Div(dash_table.DataTable(id="data_top_rows_table"), id="data_top_rows_div")
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "selected_data": misc.HtmlElement("data", "value"),
    "selected_columns": misc.HtmlElement("data_column_selector", "value"),
}

outputs = [Output("data_top_rows_div", "children"), Output("data_column_selector_div", "children"),
           Output("data_top_rows", "children")]


@cache.memoize()
def get_data(data_source):
    logger.debug(f'Trying to get data source: {data_source}')
    df = data_sources[data_source]() if data_source is not None else None

    #logger.debug(f'Data read:\n{df}.')
    return df


@cache.memoize()
def get_selected_columns(df, selected_columns):
    logger.debug(f'selected_columns: {selected_columns}')
    if df is not None and selected_columns is not None and len(selected_columns) > 0:
        return text_processing.join_columns(df, selected_columns)


def get_data_selection_output(selected_data):
    df = get_data(selected_data)
    top_rows_text = f"Cabeçalho ({0 if df is None else len(df)} histórias de usuários)"

    return misc.generate_datatable(df, "data_top_rows_table", 5), \
           misc.generate_column_picker(df, 'data_column_selector'), top_rows_text
