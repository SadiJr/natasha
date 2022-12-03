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

data_to_array_dropdown = misc.DropdownWithOptions(
    header="Opções de Vetorização dos Dados:", dropdown_id="to_array", dropdown_objects={
        "TFIDF": text_processing.TFIDF
    }, include_refresh_button=True
)

data_to_array_tab = dcc.Tab(
    label="Representação dos Dados", children=[
        html.Div(id="data_to_array_area", children=[
            data_to_array_dropdown.generate_dash_element(),
            html.H5("Data array:", id="data_to_array_header"),
            html.Div(dash_table.DataTable(id="data_to_array_table"), id="data_to_array_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "data_to_array_method": misc.HtmlElement(*data_to_array_dropdown.dropdown_args),
    "data_to_array_options": misc.HtmlElement(*data_to_array_dropdown.options_args),
    "data_to_array_refresh": misc.HtmlElement(*data_to_array_dropdown.refresh_args),
    "data_to_array_table": misc.HtmlElement("data_to_array_div", "children"),
}
outputs = [Output("data_to_array_div", "children"), Output("data_to_array_header", "children")]


@cache.memoize()
def get_data_as_array(df, data_to_array_method, data_to_array_options):
    logger.debug(f'data_to_array_method: {data_to_array_method}\ndata_to_array_options: {data_to_array_options}')
    df_arr = None
    if df is not None and data_to_array_options:
        df_arr = data_to_array_dropdown.apply(data_to_array_method, data_to_array_options, df)

    #logger.debug(f'Vectorized data:\n{df_arr}')
    return df_arr


def get_data_to_array_output(df, data_to_array_method, data_to_array_options):
    logger.debug(f'\ndata_to_array_method: {data_to_array_method}\ndata_to_array_options: {data_to_array_options}')
    df_arr = get_data_as_array(df, data_to_array_method, data_to_array_options)

    data_to_array_header = f'Vetor Gerado {(0, 0) if df_arr is None else df_arr.shape}:'
    df_sample = df_arr.sample(min(df_arr.shape[1], 20), axis=1).round(5) if df_arr is not None else None

    return misc.generate_datatable(df_sample, "data_to_array", 10, max_cell_width=None),  data_to_array_header
