import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting as bpl
from math import pi
from bokeh.models import Div, ColumnDataSource, ColorBar
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Spectral6, Viridis256 as colors
import seaborn as sns
import random
import panel.widgets as pnw
 # Import the necessary palettes
from bokeh.palettes import Category20
from bokeh.models import HoverTool

from bokeh.models import ColorBar, LinearColorMapper

# Read the dataset into a Pandas DataFrame
#df = pd.read_csv('dataset.csv')

def createApp1():
    df1 = pd.read_csv('dataset.csv', index_col=0)
    # Create an empty DataFrame to store the sampled data
    df = pd.DataFrame()

    # Group the original DataFrame by 'track_genre'
    grouped = df1.groupby('track_genre')

    # Iterate over each group, and sample 20% of the data for each 'track_genre'
    for genre, group_df in grouped:
        df = pd.concat([df, group_df.sample(frac=0.05, random_state=42)])

    # Reset the index of the sampled data
    df.reset_index(drop=True, inplace=True)


    def genre_distribution():
        genre_counts = df['track_genre'].value_counts()
        genre_distribution_plot = bpl.figure(
            title='Track Genre Distribution',
            x_axis_label='Genre',
            y_axis_label='Count',
            x_range=list(genre_counts.index),
            sizing_mode='stretch_width',
            tools='hover',  # Enable hover tool
            tooltips=[("Genre", "@x")],  # Define tooltips
        )
        genre_distribution_plot.vbar(x=list(genre_counts.index), top=genre_counts.values, width=0.8, fill_color='#68B984')

        return genre_distribution_plot

    # Define a function to create a Panel plot for popularity
    def popularity_distribution(selected_genre=None):
        plot = bpl.figure(
            title=f'Track Popularity Distribution for {selected_genre or "All Genres"}',
            x_axis_label='Popularity',
            y_axis_label='Frequency',
            sizing_mode='stretch_width'
        )

        if selected_genre:
            # Filter the DataFrame based on the selected genre
            filtered_df = df[df['track_genre'] == selected_genre]
            hist, edges = np.histogram(filtered_df['popularity'], bins=20)
        else:
            # If no genre is selected, use the entire dataset
            hist, edges = np.histogram(df['popularity'], bins=20)

        plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#004225", line_color="white")

        return plot




    # Define a function to create a Panel plot for key distribution
    def key_distribution(selected_genre=None):
        if selected_genre:
            key_counts = df[df['track_genre'] == selected_genre]['key'].value_counts()
            key_counts = key_counts.sort_index()
        else:
            key_counts = df['key'].value_counts()
            key_counts = key_counts.sort_index()

        key_distribution_plot = bpl.figure(
            title=f'Key Distribution for {selected_genre or "All Genres"}',
            x_axis_label='Key',
            y_axis_label='Count',
            x_range=[str(key) for key in key_counts.index],
            sizing_mode='stretch_width'
        )
        key_distribution_plot.vbar(x=[str(key) for key in key_counts.index], top=key_counts.values, width=0.8, fill_color='#186F65')
        
        return key_distribution_plot



    # Define a function to create a Panel plot for time signature distribution
    def time_signature_distribution(selected_genre=None):
        if selected_genre:
            time_sig_counts = df[df['track_genre'] == selected_genre]['time_signature'].value_counts()
            time_sig_counts = time_sig_counts.sort_index()
        else:
            time_sig_counts = df['time_signature'].value_counts()
            time_sig_counts = time_sig_counts.sort_index()

        time_sig_distribution_plot = bpl.figure(
            title=f'Time Signature Distribution for {selected_genre or "All Genres"}',
            x_axis_label='Time Signature',
            y_axis_label='Count',
            x_range=[str(ts) for ts in time_sig_counts.index],
            sizing_mode='stretch_width'
        )
        time_sig_distribution_plot.vbar(x=[str(ts) for ts in time_sig_counts.index], top=time_sig_counts.values, width=0.8, fill_color='#A8DF8E')

        
        return time_sig_distribution_plot



    def explicit_distribution(selected_genre=None):
        if selected_genre:
            # Filter the DataFrame based on the selected genre
            filtered_df = df[df['track_genre'] == selected_genre]
            explicit_counts = filtered_df['explicit'].value_counts()
        else:
            explicit_counts = df['explicit'].value_counts()

        explicit_labels = ["Explicit", "Not Explicit"]
        explicit_values = explicit_counts.tolist()
        
        # Calculate the percentages
        total_count = sum(explicit_values)
        percentages = [count / total_count * 100 for count in explicit_values]

        explicit_distribution_plot = bpl.figure(
            title=f'Explicit Tracks Distribution for {selected_genre or "All Genres"}',
            sizing_mode='stretch_width'
        )

        # Define a smaller radius for the pie chart
        radius = 0.2  # You can adjust this value to make the pie chart smaller

        # Calculate start and end angles for explicit and non-explicit wedges
        start_angle = 0
        end_angle_explicit = start_angle + percentages[0] / 100 * 2 * pi
        start_angle_non_explicit = end_angle_explicit
        end_angle_non_explicit = start_angle_non_explicit + percentages[1] / 100 * 2 * pi

        # Add explicit and non-explicit wedges with percentage labels
        explicit_distribution_plot.wedge(x=0, y=1, radius=radius,
            start_angle=start_angle, end_angle=end_angle_explicit,
            line_color="white", fill_color="red", legend_label=f"Explicit: {percentages[0]:.2f}%")
        explicit_distribution_plot.wedge(x=0, y=1, radius=radius,
            start_angle=start_angle_non_explicit, end_angle=end_angle_non_explicit,
            line_color="white", fill_color="green", legend_label=f"Not Explicit: {percentages[1]:.2f}%")
        explicit_distribution_plot.axis.axis_label = None
        explicit_distribution_plot.axis.visible = False
        explicit_distribution_plot.grid.grid_line_color = None

        return explicit_distribution_plot




    # Define a function to create a Box Plot of Danceability by Genre
    def danceability_box_plot(selected_genre=None):
        if selected_genre:
            # Filter the DataFrame based on the selected genre
            genre_df = df[df['track_genre'] == selected_genre]
        else:
            # If no genre is selected, use all genres
            genre_df = df

        genre_names = list(genre_df['track_genre'].unique())
        genre_names.sort()
        
        source = ColumnDataSource(genre_df)

        p = bpl.figure(
            title=f'Danceability for {selected_genre or "All Genres"}',
            x_range=genre_names,
            y_axis_label='Danceability',
            height=400,
            sizing_mode='stretch_width'
        )

        p.vbar(x='track_genre', top='danceability', width=0.7, source=source, line_color="white",
            fill_color=factor_cmap('track_genre', palette=Spectral6, factors=genre_names))

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.y_range.end = 1

        hover = HoverTool()
        hover.tooltips = [("Genre", "@track_genre"), ("Danceability", "@danceability")]
        p.add_tools(hover)

        return p


      
    def create_correlation_heatmap():
        audio_feature_columns = ['energy', 'danceability', 'valence', 'acousticness', 'instrumentalness', 'speechiness']

        # Calculate the correlation matrix for audio features
        audio_features = df[audio_feature_columns]
        corr = audio_features.corr()

        # Create a colormap using the LinearColorMapper
        mapper = LinearColorMapper(palette=list((colors)), low=-1, high=1)

        # Create a Bokeh figure for the correlation heatmap
        correlation_heatmap_figure = bpl.figure(
            title="Correlation Heatmap",
            x_range=list(corr.index),
            y_range=list((corr.index)), 
            width=600,
            height=600,
            tools="hover",
            tooltips=[('Correlation', '@image')],
            sizing_mode="fixed"
        )

        # Create an image renderer for the correlation matrix
        correlation_heatmap_figure.image(
            image=[corr.values],  # Convert the correlation matrix to a 2D array
            x=0,
            y=0,
            dw=corr.shape[1],
            dh=corr.shape[0],
            color_mapper=mapper,
        )

        # Add color bar
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
        correlation_heatmap_figure.add_layout(color_bar, 'right')

        # Set axis labels
        correlation_heatmap_figure.xaxis.major_label_orientation = np.pi / 4
        correlation_heatmap_figure.xaxis.major_label_text_font_size = "10pt"
        correlation_heatmap_figure.yaxis.major_label_text_font_size = "10pt"


        # Create a Panel plot for the correlation heatmap
        correlation_heatmap_pane = pn.pane.Bokeh(correlation_heatmap_figure, sizing_mode="stretch_both")

        return correlation_heatmap_pane
    

    def scatter_plot(source, x, y, title, x_label, y_label):
        p = bpl.figure(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            sizing_mode='stretch_width',
            tools="pan,box_zoom,reset"
        )
        
        p.scatter(x, y, source=source, size=8, fill_color='#1DB954',line_color="white")

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
            genre_df = df
        
        title = f'Danceability vs. Energy for {selected_genre or "All Genres"}'
        x_label = 'Energy'
        y_label = 'Danceability'
        
        tooltips = [("Energy", "@energy"), ("Danceability", "@danceability")]
        
        source = ColumnDataSource(genre_df)

        return scatter_plot(source, 'energy', 'danceability', title, x_label, y_label)


    def popularity_vs_acousticness(selected_genre=None):
        if selected_genre:
            genre_df = df[df['track_genre'] == selected_genre]
        else:
            genre_df = df
        
        title = f'Popularity vs. Acousticness for {selected_genre or "All Genres"}'
        x_label = 'Acousticness'
        y_label = 'Popularity'

        source = ColumnDataSource(genre_df)

        
        return scatter_plot(source, 'acousticness', 'popularity', title, x_label, y_label)



    def get_top_genres_by_attribute(attribute, top_n=10):
        # Group the data by genre and calculate the mean of the selected attribute
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
        # Sort the DataFrame by the selected attribute in descending order
        #sorted_df = df.sort_values(by=attribute, ascending=False)
        # Get the top N artists
        top_artists = df.groupby('artists', as_index=False)[attribute].mean().nlargest(top_n, attribute)
        return top_artists


    def top_artists_by_attribute(selected_attribute='popularity', top_n=10):
        top_artists = get_top_artists_by_attribute(selected_attribute, top_n)

        p = bpl.figure(
            y_range=top_artists['artists'][::-1],  # Reverse the order to display the top artists at the top
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
        
        return top_artists_by_genre



    # Function to create a table with the top artists for each genre
    def top_artists_by_genre_table(selected_genre=None, top_n=10):
        top_artists_by_genre = get_top_artists_by_genre(top_n)
        table_data = []

        if selected_genre is None or selected_genre == 'All Genres':
            for genre, top_artists in top_artists_by_genre.items():
                table_data.append((genre, top_artists.index.tolist()))
        elif selected_genre in top_artists_by_genre:
            top_artists = top_artists_by_genre[selected_genre]
            table_data.append((selected_genre, top_artists.index.tolist()))
        else:
            table_data.append(('No Genre Selected', []))

        table_data = pd.DataFrame(table_data, columns=['Genre', 'Top Artists'])
        table = pn.widgets.DataFrame(table_data,widths={'Top Artists': 400})
        
        return table


    
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



    data_table = pnw.DataFrame(
    df,
    width=1300,
    min_height=400,
    styles={
        'font-size': '14px',  # Example style attribute
    },
    sizing_mode="stretch_width",
    autosize_mode="force_fit",
)


    genre_selector = pn.widgets.MultiChoice(name='Genres', options=['All Genres'] + df['track_genre'].unique().tolist(), width=200)
    refresh_button = pn.widgets.Button(name="Refresh Charts",button_type='success')
    top_artists_or_genres_attribute_selector = pn.widgets.Select(
    name='Select Attribute',
    options=['popularity', 'danceability', 'energy', 'valence', 'acousticness','loudness','liveness','tempo'],
    value='popularity',
    width=200)


    # Define a callback function to update the charts when the genre selection changes
    def update_charts(event):
        selected_genres = event.new
        if selected_genres and selected_genres[0] != 'All Genres':
            selected_genre = selected_genres[0]  # Use the first selected genre
        else:
            selected_genre = None  # If 'All Genres' is selected, set to None
        popularity_chart.object = popularity_distribution(selected_genre)
        key_distribution_chart.object = key_distribution(selected_genre)
        time_signature_distribution_chart.object = time_signature_distribution(selected_genre)
        explicit_distribution_chart.object = explicit_distribution(selected_genre)
        danceability_chart.object=danceability_box_plot(selected_genre)
        danceability_energy_chart.object = danceability_vs_energy(selected_genre)
        popularity_acousticness_chart.object = popularity_vs_acousticness(selected_genre)
        top_artists_by_genre_table_chart.object = top_artists_by_genre_table(selected_genre)


    # Define a callback function to update the chart
    def update_top_artists_or_genres_chart(event):
        selected_attribute = top_artists_or_genres_attribute_selector.value
        top_artists_or_genres_chart.object = top_genres_by_attribute(selected_attribute)
        top_artists_by_attribute_chart.object = top_artists_by_attribute(selected_attribute)
        
    

    

    genre_distribution_chart = pn.panel(genre_distribution())
    # Create a placeholder chart for popularity distribution
    popularity_chart = pn.panel(popularity_distribution())
    # Create a placeholder chart for key distribution
    key_distribution_chart = pn.panel(key_distribution())
    # Create a placeholder chart for time signature distribution
    time_signature_distribution_chart = pn.panel(time_signature_distribution())
    # Create a placeholder chart for explicit distribution (pie chart)
    explicit_distribution_chart = pn.panel(explicit_distribution())
    danceability_chart=pn.panel(danceability_box_plot())
    # Call the function to create the correlation heatmap
    correlation_heatmap_pane = create_correlation_heatmap()
    danceability_energy_chart = pn.panel(danceability_vs_energy(), sizing_mode="stretch_width")
    popularity_acousticness_chart = pn.panel(popularity_vs_acousticness(), sizing_mode="stretch_width")
    top_artists_or_genres_chart = pn.panel(top_genres_by_attribute())
    top_artists_by_attribute_chart = pn.panel(top_artists_by_attribute())  # Call the function to create the chart
    top_artists_by_genre_table_chart=pn.panel(top_artists_by_genre_table())





    # Add the callback to the genre_selector
    genre_selector.param.watch(update_charts, 'value')

    top_artists_or_genres_attribute_selector.param.watch(update_top_artists_or_genres_chart, 'value')
   
    # Store the initial chart states
    initial_popularity_chart = popularity_chart.object
    initial_key_distribution_chart = key_distribution_chart.object
    initial_time_signature_distribution_chart = time_signature_distribution_chart.object
    initial_explicit_distribution_chart = explicit_distribution_chart.object
    initial_danceability_chart= danceability_chart.object
    initial_danceability_energy_chart=danceability_energy_chart.object
    initial_popularity_acousticness_chart=popularity_acousticness_chart.object

   



    # Create a refresh button and define a callback function
    def refresh_charts(event):
        # Reset the charts to their initial state
        popularity_chart.object = initial_popularity_chart
        key_distribution_chart.object = initial_key_distribution_chart
        time_signature_distribution_chart.object = initial_time_signature_distribution_chart
        explicit_distribution_chart.object = initial_explicit_distribution_chart
        danceability_chart.object = initial_danceability_chart
        danceability_energy_chart.object=initial_danceability_energy_chart
        popularity_acousticness_chart.object=initial_popularity_acousticness_chart

    refresh_button.on_click(refresh_charts)

    

    # Create a sidebar with the widgets
    sidebar = pn.Column(
        pn.Column(genre_selector,
        refresh_button,
    
        
        )
        
    )

    sidebar_second = pn.Column(
        pn.Column( top_artists_or_genres_attribute_selector,
    
        
        )
        
    )

        # Create the main layout
    app_layout = pn.Column(
        description,
        data_table,
        genre_distribution_chart,
        pn.Row(
            sidebar,
            pn.Column(
                pn.Tabs(
                    ('Popularity Distribution', popularity_chart),
                    ('Key Distribution', key_distribution_chart),
                    ('Time Signature Distribution', time_signature_distribution_chart),
                    ('Explicit Tracks Distribution', explicit_distribution_chart),
                    # ('Danceability by Genre', danceability_chart),
                ),
                danceability_chart,
                pn.Row(
                    danceability_energy_chart,
                    popularity_acousticness_chart,
                ),
                top_artists_by_genre_table_chart,
            
            ),
            
        ),
        pn.Row(
            sidebar_second,
            pn.Column(
                pn.Row(
                top_artists_or_genres_chart,
                top_artists_by_attribute_chart,),

                correlation_heatmap_pane,
            ),

        

        ),
        
        
        sizing_mode="stretch_width",
    )
    custom_css = """
    .custom-navbar {
        padding: 10px;
    }

    """

    pn.config.raw_css.append(custom_css)

    return app_layout.servable()    
