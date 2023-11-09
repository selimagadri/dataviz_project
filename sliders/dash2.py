import pandas as pd
import panel as pn
from sklearn.cluster import KMeans
import holoviews as hv
import hvplot.pandas
from sklearn.preprocessing import StandardScaler

def createApp2():
    # Load your Spotify tracks dataset
    df = pd.read_csv('dataset.csv')

    # Select relevant features for clustering
    features = df[['danceability', 'energy', 'tempo', 'loudness', 'valence','speechiness','acousticness','liveness']]

    # Standardize features if necessary
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Create Panel widgets for selecting features and number of clusters
    cols = list(features.columns)
    x = pn.widgets.Select(name='x', options=cols)
    y = pn.widgets.Select(name='y', options=cols, value=cols[1])
    n_clusters = pn.widgets.IntSlider(name='n_clusters', start=1, end=10, value=3)

    def get_clusters(x, y, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(features)
        df['labels'] = kmeans.labels_.astype(str)
        centers = df.groupby('labels')[[x, y]].mean()
        return (
            df.sort_values('labels').hvplot.scatter(
                x, y, c='labels', size=100, height=500, responsive=True
            ) *
            centers.hvplot.scatter(
                x, y, marker='x', c='black', size=400, padding=0.1, line_width=5
            )
        )

    layout = pn.Row(
        pn.Column(
            '# Spotify Track Clustering',
            "This app provides an example of *building a simple dashboard using Panel*.\n\nIt demonstrates how to perform *k-means clustering on your Spotify tracks dataset*.\n\nThe entire clustering and plotting pipeline is expressed as a *single reactive function* \n\nthat responsively returns an updated plot when one of the widgets changes.",
            "The *x marks the center* of the cluster.",
            x, y, n_clusters,
        ),
        pn.Column(
            pn.pane.HoloViews(
                pn.bind(get_clusters, x, y, n_clusters), sizing_mode='stretch_width'
            )
        ),
    )

    return layout


