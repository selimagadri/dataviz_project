import pandas as pd
import panel as pn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Select
from bokeh.palettes import viridis
from bokeh.transform import factor_cmap
from colorsys import hls_to_rgb

def createApp3():
    df1 = pd.read_csv('dataset.csv', index_col=0)
    df = pd.DataFrame()

    grouped = df1.groupby('track_genre')

    for genre, group_df in grouped:
        df = pd.concat([df, group_df.sample(frac=0.01, random_state=42)])

    df.reset_index(drop=True, inplace=True)

    X = df.drop(['track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)

    def perform_pca(data, n_components=2):
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        return pca_result

    def perform_tsne(data, n_components=2, perplexity=30, learning_rate=200):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
        tsne_result = tsne.fit_transform(data)
        return tsne_result

    def perform_umap(data, n_components=2, n_neighbors=15, min_dist=0.1):
        umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap_model.fit_transform(data)
        return umap_result

    def generate_green_opposite_colors(num_colors):
        distinct_colors = []
        for i in range(num_colors):
            green_hue = i / num_colors
            opposite_hue = (green_hue + 0.5) % 1.0
            rgb_green = hls_to_rgb(green_hue, 0.5, 1.0)
            rgb_opposite = hls_to_rgb(opposite_hue, 0.5, 1.0)

            distinct_colors.extend([
                '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb_green),
                '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb_opposite)
            ])
        return distinct_colors

    # Assuming you want 114 distinct colors
    distinct_colors = generate_green_opposite_colors(114)

    # Create a Panel widget for algorithm selection with 'PCA' as the default value
    algorithm_selector = Select(title='Select Algorithm', options=['PCA', 't-SNE', 'UMAP'], value='PCA', width=200)

    # Create empty Bokeh plots with default data
    scatter_plot = figure(title="Scatter Plot", width=800, height=500)
    scatter_source = ColumnDataSource(data={'x': [], 'y': [], 'track_genre': []})

    scatter_plot.circle(x='x', y='y', source=scatter_source, size=8,
                        color=factor_cmap('track_genre', palette=distinct_colors, factors=df['track_genre'].unique()))

    # Title and Description
    title_div = pn.pane.Markdown("# PCA Scatter Plot", style={'color': '#1DB954', 'font-size': '20px'})
    description_div = pn.pane.Markdown(
        "This is a scatter plot based on Principal Component Analysis (PCA). "
        "PCA is a dimensionality reduction technique that identifies patterns in data and represents it in a lower-dimensional space.",
        style={'font-size': '14px'}
    )

    # Define a callback function to update the scatter plot based on the selected algorithm
    def update_plot(attr, old, new):
        algorithm = algorithm_selector.value
        if algorithm == 'PCA':
            result = perform_pca(X)
            title_div.object = "# PCA Scatter Plot"
            description_div.object = (
                """This is a scatter plot based on *Principal Component Analysis (PCA)*. 
                PCA is a dimensionality reduction technique that identifies patterns in data and represents it in a lower-dimensional space."""
            )
        elif algorithm == 't-SNE':
            result = perform_tsne(X)
            title_div.object = "# t-SNE Scatter Plot"
            description_div.object = (
                """This is a scatter plot based on *t-Distributed Stochastic Neighbor Embedding (t-SNE)*. 
                t-SNE is a non-linear dimensionality reduction technique particularly effective for visualization of high-dimensional data."""
            )
        elif algorithm == 'UMAP':
            result = perform_umap(X)
            title_div.object = "# UMAP Scatter Plot"
            description_div.object = (
                """This is a scatter plot based on *Uniform Manifold Approximation and Projection (UMAP)*. 
                UMAP is a dimensionality reduction technique known for preserving both local and global structure in the data."""
            )
        else:
            result = None

        if result is not None:
            scatter_source.data = {'x': result[:, 0], 'y': result[:, 1], 'track_genre': df['track_genre'].values}

    # Set the initial plot to PCA
    update_plot(None, None, None)

    # Attach the callback function to the algorithm_selector
    algorithm_selector.on_change('value', update_plot)

    # Create a Panel app layout
    app_layout = pn.Row(
        algorithm_selector,
        pn.Column(title_div, description_div, scatter_plot))

    return app_layout


