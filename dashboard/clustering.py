from dash import dcc
from dash import html
from dash.dependencies import Output

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score

import model.preprocess

import dashboard.app_misc as misc
from dashboard.cache import cache

from model.logger import LoggerFactory
from model.utils import full_stack

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

clustering_dropdown = misc.DropdownWithOptions(
    header="Escolha o tipo de agrupamento:", dropdown_id="clustering", dropdown_objects={
        "Agrupar por personas encontradas": model.clustering.Personas,
        "KMeans": model.clustering.KMeans,
        "DBSCAN": model.clustering.DBSCAN,
        "AgglomerativeClustering": model.clustering.AgglomerativeClustering,
        "GaussianMixture": model.clustering.GaussianMixture,
    }, include_refresh_button=True
)

clustering_tab = dcc.Tab(
    label="Agrupamento", children=[
        html.Div(id="clustering_area", children=[
            clustering_dropdown.generate_dash_element(),
        ]),
        html.P(children=None, id="cluster_info_text", style={"padding": "5px", "margin": "5px",  'whiteSpace': 'break-spaces'})
    ], className="custom-tab", selected_className="custom-tab--selected"
)

clusters_tab = dcc.Tab(
    label="Grupos", children=[html.Div(id="cluster_info_table")],
    className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "clustering_method": misc.HtmlElement(*clustering_dropdown.dropdown_args),
    "clustering_options": misc.HtmlElement(*clustering_dropdown.options_args),
    "clustering_refresh": misc.HtmlElement(*clustering_dropdown.refresh_args),
    "cluster_info_table": misc.HtmlElement("cluster_info_table", "children"),
}
outputs = [Output("cluster_info_table", "children"), Output("cluster_info_text", "children")]


@cache.memoize()
def get_clusters(df, to_cluster, clustering, clustering_options):
    logger.debug(f'Trying to generate clusters using options:clustering: '
                 f'{clustering}\nclustering_options: {clustering_options}.')
    if clustering_options:
        clusters = clustering_dropdown.apply(clustering, clustering_options, to_cluster)
    elif clustering == 'Agrupar por personas encontradas':
        clusters = clustering_dropdown.apply(clustering, clustering_options, to_cluster).user_name
    elif to_cluster is not None:
        clusters = np.zeros(to_cluster.shape[0])
    else:
        clusters = None

    logger.debug(f'Generated clusters:\n{clusters}.')
    return clusters


def get_cluster_info_df(n_cluster_info, clusters, titles, bow):
    """Returns dataframe with basic information regarding clusters.

    Will contain columns with cluster number, cluster size, random samples and top words.
    """
    cluster_info_rows = []

    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue

        samples = titles.loc[idx].sample(min(n_cluster_info, idx.sum()), replace=False).values
        samples = np.pad(samples, (0, max(0, n_cluster_info - len(samples))))

        top_words = bow.columns[bow.loc[idx].sum(0).argsort()[::-1][:n_cluster_info]]
        top_words = np.pad(top_words, (0, max(0, n_cluster_info - len(top_words))))

        try:
            cluster_info_rows.append([int(cluster), idx.sum(), *samples, *top_words])
        except Exception as e:
            logger.error(full_stack())

        cluster_info_rows.append([i, idx.sum(), *samples, *top_words])

    cluster_info_df = pd.DataFrame(cluster_info_rows, columns=[
        "Grupo", "Tamanho",
        *["Exemplo %d" % i for i in range(1, n_cluster_info + 1)],
        *["Palavra mais frequente %d" % i for i in range(1, n_cluster_info + 1)],
    ])

    #logger.debug(f'Generated cluster info:\n{cluster_info_df}.')
    return cluster_info_df


def get_clustering_cluster_output(df, df_arr, clustering_method, clustering_options, titles, bow):

    logger.debug(f'clustering_method: {clustering_method}\n'
          f'clustering_options: {clustering_options}')
    clusters = get_clusters(df, df_arr, clustering_method, clustering_options)

    cluster_info_df = None
    if clusters is not None and titles is not None and bow is not None:
        cluster_info_df = get_cluster_info_df(10, clusters, titles, bow)

    cluster_info_score = None
    if np.unique(clusters).size > 1:
        cluster_info_score = f"Silhouette Score: {silhouette_score(df_arr.values, clusters, random_state=42)}\n\n" \
                             f"Calinski Harabasz Score: {calinski_harabasz_score(df_arr.values, clusters)}\n\n" \
                             f"Davies Bouldin Score: {davies_bouldin_score(df_arr.values, clusters)}"

    return misc.generate_datatable(cluster_info_df, "cluster_info", 1000, "600px"), cluster_info_score
