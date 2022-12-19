from dash import dcc, html
from dash.dependencies import Output

import numpy as np
from sklearn.metrics.cluster import calinski_harabasz_score, silhouette_score, davies_bouldin_score

import model.preprocess
from model.logger import LoggerFactory

import dashboard.miscellaneous as misc
from dashboard.cache import cache

logger = LoggerFactory.get_logger(__name__, 'DEBUG')

clustering_dropdown = misc.Dropdown(
    header="Escolha o tipo de agrupamento:", dropdown_id="clustering", dropdown_objects={
        "Agrupar por personas encontradas": model.clustering.Personas,
        "K-Means": model.clustering.KMeans,
        "DBSCAN": model.clustering.DBSCAN,
        "Hierárquico Aglomerativo": model.clustering.AgglomerativeClustering,
        "Gaussian Mixture Models": model.clustering.GMM,
    }, include_refresh_button=True
)

clustering_tab = dcc.Tab(
    label="Agrupamento", children=[
        html.Div(id="clustering_area", children=[
            clustering_dropdown.generate_dash_element(),
        ]),
        html.P(children=None, id="cluster_info_text", style={"padding": "5px", "margin": "5px", 'whiteSpace': 'break-spaces'})
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "clustering_method": misc.HtmlElement(*clustering_dropdown.dropdown_args),
    "clustering_options": misc.HtmlElement(*clustering_dropdown.options_args),
    "clustering_refresh": misc.HtmlElement(*clustering_dropdown.refresh_args),
}
outputs = Output("cluster_info_text", "children")


@cache.memoize()
def get_clusters(df, to_cluster, clustering, clustering_options):
    logger.debug(f'Trying to generate clusters using options:clustering: '
                 f'{clustering}\nclustering_options: {clustering_options}.')
    if clustering_options:
        clusters = clustering_dropdown.apply(clustering, clustering_options, to_cluster)
    elif clustering == 'Agrupar por personas encontradas':
        clusters = clustering_dropdown.apply(clustering, clustering_options, df).user_name
    elif to_cluster is not None:
        clusters = np.zeros(to_cluster.shape[0])
    else:
        clusters = None

    logger.debug(f'Generated clusters:\n{clusters}.')
    return clusters


def get_clustering_cluster_output(df, df_arr, clustering_method, clustering_options, titles):

    logger.debug(f'clustering_method: {clustering_method}\n'
          f'clustering_options: {clustering_options}')
    clusters = get_clusters(df, df_arr, clustering_method, clustering_options)

    cluster_info_score = None
    if np.unique(clusters).size > 1:
        cluster_info_score = f"Silhouette Score: {silhouette_score(df_arr.values, clusters, random_state=42)}\n\n" \
                             f"Calinski Harabasz Score: {calinski_harabasz_score(df_arr.values, clusters)}\n\n" \
                             f"Davies Bouldin Score: {davies_bouldin_score(df_arr.values, clusters)}"

    return cluster_info_score
