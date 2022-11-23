from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Output

import pandas as pd

import model

import dashboard.app_misc as misc
from dashboard.cache import cache

from model.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

dim_reduction_dropdown = misc.DropdownWithOptions(
    header="(Opcional) Reduza a dimensionalidade dos dados antes de agrupar:", dropdown_id="dim_reduction",
    dropdown_objects={
        "PCA": model.dim_reduction.PCA,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)

dim_reduction_tab = dcc.Tab(
    label="Redução de Dimensionalidade", children=[
        html.Div(id="dim_red_area", children=[
            dim_reduction_dropdown.generate_dash_element(),
            html.H5("Vetor Reduzido:", id="dim_red_table_header"),
            html.Div(dash_table.DataTable(id="dim_red_table"), id="dim_red_table_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "dim_reduction_method": misc.HtmlElement(*dim_reduction_dropdown.dropdown_args),
    "dim_reduction_options": misc.HtmlElement(*dim_reduction_dropdown.options_args),
    "dim_reduction_refresh": misc.HtmlElement(*dim_reduction_dropdown.refresh_args),
}
outputs = Output("dim_red_table_div", "children")


@cache.memoize()
def get_dim_reduction(df_arr, dim_reduction_method, dim_reduction_options):
    logger.debug(f'dim_reduction_method: {dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}')
    if df_arr is not None and dim_reduction_method and dim_reduction_options:
        return pd.DataFrame(dim_reduction_dropdown.apply(dim_reduction_method, dim_reduction_options, df_arr))

    return None


def get_dim_reduction_output(df_arr, dim_reduction_method, dim_reduction_options):
    logger.debug(f'dim_reduction_method: {dim_reduction_method}\ndim_reduction_options: {dim_reduction_options}')
    df_arr_dim_red = get_dim_reduction(df_arr, dim_reduction_method, dim_reduction_options)

    sample_df = None
    if df_arr_dim_red is not None:
        sample_df = df_arr_dim_red.sample(min(df_arr_dim_red.shape[1], 20), axis=1).round(2)

    return misc.generate_datatable(sample_df, "dim_red_table", 5, max_cell_width="350px")
