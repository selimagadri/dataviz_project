import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting as bpl
from math import pi
from bokeh.models import Div, ColumnDataSource, ColorBar,  LinearColorMapper, NumeralTickFormatter
from bokeh.palettes import Greens, Viridis256 as colors
from bokeh.models import HoverTool

def createApp1():
    df = pd.read_csv('dataset.csv', index_col=0)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['track_id',  'album_name', 'track_name'], inplace=True)

    def genre_distribution():
        genre_counts = df['track_genre'].value_counts()

        genre_distribution_plot = bpl.figure(
            title='Track Genre Distribution',
            x_axis_label='Genre',
            y_axis_label='Count',
            x_range=list(genre_counts.index),
            sizing_mode='stretch_width',
            tools='hover',
            tooltips=[("Genre", "@x")],
        )

        genre_distribution_plot.xaxis.major_label_orientation = np.pi / 4
        genre_distribution_plot.xaxis.major_label_text_font_size = "7pt"
        genre_distribution_plot.yaxis.major_label_text_font_size = "7pt"
        
        genre_distribution_plot.vbar(x=list(genre_counts.index), top=genre_counts.values, width=0.8, fill_color='#68B984')

        return genre_distribution_plot
    
    def create_histogram_plot(df, selected_genre, selected_column):
        filtered_df = df if selected_genre == "All Genres" else df[df['track_genre'] == selected_genre]
        hist_plot = filtered_df[selected_column].hvplot.hist(
            bins=10, color='lightblue', alpha=0.7, height=400, width=400, 
            title=f'Histogram of {selected_column.title()} for {selected_genre}'
        )
        return hist_plot

    def create_box_plot(df, selected_genre, selected_column):
        filtered_df = df if selected_genre == "All Genres" else df[df['track_genre'] == selected_genre]
        box_plot = filtered_df[selected_column].hvplot.box(
            height=400, width=400, title=f'Box plot of {selected_column} for {selected_genre}'
        )
        return box_plot



    def create_kde_plot(df, selected_genre, selected_column):
        filtered_df = df if selected_genre == "All Genres" else df[df['track_genre'] == selected_genre]
        kde_plot = filtered_df[selected_column].hvplot.kde(
            fill_alpha=0.5, line_width=2, color='purple', height=400, width=400, 
            title=f'KDE Plot of {selected_column} for {selected_genre}',
            hover = False
        )
        return kde_plot



    def scatter_plot(source, x, y, title, x_label, y_label):
        p = bpl.figure(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            sizing_mode='stretch_width',
            tools="pan,box_zoom,reset"
        )
        
        p.scatter(x, y, source=source, size=8, fill_color='#1DB954', line_color="#1DB954")

        hover = HoverTool()
        hover.tooltips = [
            (x_label, f'@{x}'),
            (y_label, f'@{y}')
        ]
        p.add_tools(hover)
        
        return p
    
    def danceability_vs_energy(selected_genre=None):
        if selected_genre:
            genre_df = df[df['track_genre'] == selected_genre]
        else:
            genre_df = df[df['track_genre'] == 'anime']
        
        title = f'Danceability vs. Energy for {selected_genre or " anime"}'
        x_label = 'Energy'
        y_label = 'Danceability'
        
        source = ColumnDataSource(genre_df)

        return scatter_plot(source, 'energy', 'danceability', title, x_label, y_label)
    

    def popularity_vs_acousticness(selected_genre=None):
        if selected_genre:
            genre_df = df[df['track_genre'] == selected_genre]
        else:
            genre_df = df[df['track_genre'] == 'anime']
        
        title = f'Popularity vs. Acousticness for {selected_genre or "anime"}'
        x_label = 'Acousticness'
        y_label = 'Popularity'

        source = ColumnDataSource(genre_df)

        return scatter_plot(source, 'acousticness', 'popularity', title, x_label, y_label)
    

    def create_correlation_heatmap():
        audio_feature_columns = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'speechiness']

        # Calculate the correlation matrix for audio features
        audio_features = df[audio_feature_columns]
        corr = audio_features.corr()

        custom_palette = list(Greens[9])
    
        color_mapper = LinearColorMapper(palette=custom_palette, low=corr.min().min(), high=corr.max().max())

        flat_values = corr.values.flatten()

        x_vals = [col for col in corr.columns for _ in range(len(corr))]
        y_vals = [index for _ in range(len(corr.columns)) for index in corr.index]
        color_vals = flat_values

        source = ColumnDataSource(data=dict(
            x=x_vals,
            y=y_vals,
            image=color_vals,
        ))

        heatmap_figure = bpl.figure(
            title="Correlation Heatmap",
            x_range=[str(col) for col in corr.columns],
            y_range=[str(index) for index in corr.index],
            width=600,
            height=600,
            tools="hover",
            tooltips=[('Features', '@y, @x'), ('Correlation', '@image')],
            sizing_mode="fixed"
        )

        heatmap_figure.rect(
            x='x',
            y='y',
            width=1,
            height=1,
            fill_color={'field': 'image', 'transform': color_mapper},
            line_color=None,
            source=source,
        )

        color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
        heatmap_figure.add_layout(color_bar, 'right')

        heatmap_figure.xaxis.major_label_orientation = np.pi / 4
        heatmap_figure.xaxis.major_label_text_font_size = "10pt"
        heatmap_figure.yaxis.major_label_text_font_size = "10pt"

        heatmap_pane = pn.pane.Bokeh(heatmap_figure, sizing_mode="stretch_both")

        return heatmap_pane

    

    def get_top_genres_by_attribute(attribute, top_n=10):
   
        top = df.groupby('track_genre')[attribute].mean().nlargest(top_n)
        return top
    

    def top_genres_by_attribute(selected_attribute='popularity', top_n=10):
        top_data = get_top_genres_by_attribute(selected_attribute, top_n)
        
        p = bpl.figure(
            x_range=list(top_data.index),
            title=f'Top {top_n} genres based on  {selected_attribute.capitalize()}',
            x_axis_label=selected_attribute.capitalize(),
            y_axis_label=f'Average {selected_attribute.capitalize()}',
            sizing_mode='stretch_width',
            toolbar_location=None,
        )
        
        p.vbar(x=top_data.index, top=top_data.values, width=0.7, fill_color="#03C988", line_color="#03C988")

        p.xaxis.major_label_orientation = np.pi / 4

        return p




    def get_top_artists_by_attribute(attribute, top_n=10):
   
        top_artists = df.groupby('artists', as_index=False)[attribute].mean().nlargest(top_n, attribute)
        return top_artists


    def top_artists_by_attribute(selected_attribute='popularity', top_n=10):
        top_artists = get_top_artists_by_attribute(selected_attribute, top_n)

        p = bpl.figure(
            y_range=top_artists['artists'][::-1], 
            title=f'Top {top_n} Artists based on {selected_attribute.capitalize()}',
            x_axis_label=selected_attribute.capitalize(),
            y_axis_label='Artists',
            sizing_mode='stretch_width',
            toolbar_location=None,
        )

        p.hbar(y=top_artists['artists'][::-1], right=top_artists[selected_attribute][::-1],
            height=0.7, fill_color="#68B984", line_color="#68B984")

        return p



    def get_top_artists_by_genre(top_n=10):
        top_artists_by_genre = {}
        genres = df['track_genre'].unique()

        for genre in genres:
            top_artists = df[df['track_genre'] == genre]['artists'].value_counts().nlargest(top_n)
            top_artists_by_genre[genre] = top_artists

        result_df = pd.concat(top_artists_by_genre.values(), keys=top_artists_by_genre.keys(), names=['Genre'])

        result_df = result_df.reset_index()

        return result_df

    static_table_data = get_top_artists_by_genre()
    static_table = pn.widgets.DataFrame(static_table_data, name='Top Artists by Genre', sizing_mode="stretch_width")



    
    description_div = Div(
    text="""
    <style>
        .description {
            text-align: center;
            margin-bottom: 20px;
        }
        .title {
            color: #1DB954;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .line {
            color: black;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .table {
            margin-top: 20px;
        }
    </style>
    <div class="description">
        <p class="title">Dashboard Description</p>
        <p class="line">This dashboard provides an interactive visualization of the data using Bokeh, Plotly, and Panel.</p>
        <p class="line">You can explore various aspects of the dataset using the options provided in the sidebar.</p>
    </div>
    """)


    # Create a Panel column for the dashboard description
    description = pn.Column(
        pn.Row(pn.layout.HSpacer(), description_div, pn.layout.HSpacer()),
        sizing_mode="stretch_width",
    )



    data_table = pn.widgets.DataFrame(df, name='Data Table', sizing_mode="stretch_width")



    genre_options = ["All Genres"] + df['track_genre'].unique().tolist()
    genre_selector = pn.widgets.Select(options=genre_options, name='Select Genre')
    column_selector = pn.widgets.Select(options=list(df.select_dtypes(include=["int", "float"]).columns), name='Select Column')
    refresh_button = pn.widgets.Button(name='Refresh Plots', button_type='primary')
    
    def update_plots(selected_genre, selected_column):
        hist_plot = create_histogram_plot(df, selected_genre, selected_column)
        box_plot = create_box_plot(df, selected_genre, selected_column)
        kde_plot = create_kde_plot(df, selected_genre, selected_column)

        default_genre = 'anime'
        scatter_genre = default_genre if selected_genre == "All Genres" else selected_genre

        danceability_vs_energy_plot = danceability_vs_energy(None if selected_genre == "All Genres" else selected_genre)
        popularity_vs_acousticness_plot = popularity_vs_acousticness(None if selected_genre == "All Genres" else selected_genre)
    

        top_genres_plot = top_genres_by_attribute(selected_column)
        top_artists_plot = top_artists_by_attribute(selected_column)

        
        
        return pn.Column(
            pn.Row(hist_plot, box_plot, kde_plot),
            pn.Row(danceability_vs_energy_plot, popularity_vs_acousticness_plot),
            pn.Row(top_genres_plot, top_artists_plot),
        )
    
    # Create a function to reset the plots to the initial state
    def reset_plots(event):
        # Reset the selectors to their default values
        genre_selector.value = "All Genres"
        column_selector.value = df.select_dtypes(include=["int", "float"]).columns[0]

    # Add a callback to the refresh button
    refresh_button.on_click(reset_plots)

    dashboard = pn.Column(
        description,
        data_table,
        genre_distribution(),
        pn.Row(
        pn.Column(
            genre_selector,
            column_selector,
            refresh_button,
        ),

        pn.Column(
            pn.Row(
            pn.bind(update_plots, selected_genre=genre_selector, selected_column=column_selector)),
            static_table,
            create_correlation_heatmap()
            ),
        )    
    )

    return dashboard.servable()    
