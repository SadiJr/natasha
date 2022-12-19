from dash import dcc, html
from dash.dependencies import Output

import numpy as np
import matplotlib
import pandas as pd
import plotly.graph_objs as go

import model

import dashboard.miscellaneous as misc
from dashboard.cache import cache

from model.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

plot_tab = dcc.Tab(
    label="Gráfico", children=[dcc.Graph(id="scatter-plot")],
    className="custom-tab", selected_className="custom-tab--selected"
)

plot_options = misc.Dropdown(
    header="Escolha o algoritmo de redução de dimensionalidade para exibir o gráfico:",
    dropdown_id="plot_dim_reduction", dropdown_objects={
        "PCA": model.dim_reduction.PCA,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)
arguments = {
    "plot_dim_reduction_method": misc.HtmlElement(*plot_options.dropdown_args),
    "plot_dim_reduction_options": misc.HtmlElement(*plot_options.options_args),
    "plot_dim_reduction_refresh": misc.HtmlElement(*plot_options.refresh_args),
}
outputs = Output("scatter-plot", "figure")

plot_opt_tab = dcc.Tab(
    label="Opções de Gráfico", children=[
        html.Div(id="plotting_dim_red_area", children=[
            plot_options.generate_dash_element(),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

@cache.memoize()
def get_dim_reduction(df_arr, plot_dim_reduction_method, plot_dim_reduction_options):
    logger.debug(f'plot_dim_reduction_method: {plot_dim_reduction_method}\nplot_dim_reduction_options: '
                 f'{plot_dim_reduction_options}.')
    if plot_dim_reduction_method and plot_dim_reduction_options:
        return pd.DataFrame(plot_options.apply(plot_dim_reduction_method, plot_dim_reduction_options,
                                               df_arr))

    return None

# https://stackoverflow.com/a/55828367
def plots(coords, clusters, titles):
    logger.debug(f'coords: {coords}')

    np.random.seed(80)
    colors = np.random.choice(list(matplotlib.colors.cnames.values()), size=np.unique(clusters).size, replace=True)

    dims = list(zip(("x", "y", "z"), range(coords.shape[1])))
    scatter_class = go.Scatter3d if len(dims) == 3 else go.Scatter
    scatter_plots = []
    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue

        scatter_plots.append(scatter_class(
            name=f"Grupo {cluster}",
            **{label: coords[idx, i] for label, i in dims},
            text=titles.values[idx],
            textposition="top center",
            mode="markers",
            marker=dict(size=5 if len(dims) == 3 else 12, symbol="circle", color=colors[i]),
        ))

    logger.debug(f'Generated scatter plot:\n{scatter_plots}')
    return scatter_plots


def get_plot_output(df_arr, plot_dim_reduction_method, plot_dim_reduction_options, clusters, titles):
    logger.debug(f'plot_dim_reduction_method: {plot_dim_reduction_method}\n'
                 f'plot_dim_reduction_options: {plot_dim_reduction_options}')
    if df_arr is None or not plot_dim_reduction_method or not plot_dim_reduction_options:
        return go.Figure(layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2", height=1000))

    coords_df = get_dim_reduction(df_arr, plot_dim_reduction_method, plot_dim_reduction_options)
    scatter_plots = plots(coords_df.values, clusters, titles)

    return go.Figure(data=scatter_plots, layout=go.Layout(
        margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2",
        legend={"bgcolor": "#f2f2f2"}, hovermode="closest", height=1000))
