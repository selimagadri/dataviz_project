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

def createApp2():
    df = pd.read_csv('dataset.csv')

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

        plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#0E21A0", line_color="white")

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
        key_distribution_plot.vbar(x=[str(key) for key in key_counts.index], top=key_counts.values, width=0.8, fill_color='#4D2DB7')
        
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
        time_sig_distribution_plot.vbar(x=[str(ts) for ts in time_sig_counts.index], top=time_sig_counts.values, width=0.8, fill_color='#9D44C0')

        
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



    # Define a function to create a Panel plot for attribute averages by genre
    def attribute_averages_by_genre(selected_attribute='popularity'):
        if selected_attribute not in df.columns:
            return pn.pane.Alert("Invalid Attribute", alert_type="error")

        avg_by_genre = df.groupby('track_genre')[selected_attribute].mean()

        # Convert the index (genre names) to a list of strings
        x_range = list(avg_by_genre.index)

        plot = bpl.figure(
            x_range=x_range,  # Genre names on the x-axis
            title=f'Average {selected_attribute.capitalize()} by Genre',
            x_axis_label='Genre',
            y_axis_label=f'Average {selected_attribute.capitalize()}',
            sizing_mode='stretch_width'
        )
        plot.vbar(x=x_range, top=avg_by_genre.values, width=0.8, fill_color="#7752FE", line_color="white")

        return plot


    # Define a function to create a Panel plot for popularity vs. energy
    def popularity_vs_energy(selected_track=None):
        plot = bpl.figure(
            title='Popularity vs. Energy for Selected Track',
            x_axis_label='Energy',
            y_axis_label='Popularity',
            sizing_mode='stretch_both'
        )

        if selected_track:
            track_data = df[df['track_name'] == selected_track]
            plot.circle(track_data['energy'], track_data['popularity'], size=8, fill_color='blue')
            plot.title.text = f'Popularity vs. Energy for "{selected_track}"'
            plot.title.align = 'center'
        else:
            plot.circle(df['energy'], df['popularity'], size=8, fill_color='blue')
            plot.title.text = 'Popularity vs. Energy for All Tracks'
            plot.title.align = 'center'

        

        return plot



   
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
            tooltips=[('Features', '@y, @x'), ('Correlation', '@image')],
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


    
    description = pn.Column(
    pn.Row(pn.layout.HSpacer(), pn.pane.Markdown("## Dashboard Description", styles={"font-size": "20px"}), pn.layout.HSpacer()),

    # ...

    pn.pane.Markdown(
        """
        This dashboard provides an interactive visualization of the data using Bokeh, Plotly, and Panel.
        You can explore various aspects of the dataset using the options provided in the sidebar.
        """
    ),
    # pn.layout.VSpacer(height=5),
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
    refresh_button = pn.widgets.Button(name="Refresh Charts")
    attribute_selector = pn.widgets.Select(name='Select Attribute', options=df.columns.to_list(), value='popularity', width=200)
    attribute_refresh_button = pn.widgets.Button(name="Refresh Attribute Chart")
    track_selector = pn.widgets.Select(name='Select Track', options=['All Tracks'] + df['track_name'].unique().tolist(), width=200)

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

    # Define a callback function to update the attribute chart when the selection changes
    def update_attribute_chart(event):
        selected_attribute = event.obj.value
        attribute_average_chart.object = attribute_averages_by_genre(selected_attribute)


    # Define a callback function to update the popularity vs. energy chart
    def update_popularity_energy_chart(event):
        selected_track = event.obj.value
        popularity_energy_chart.object = popularity_vs_energy(selected_track)


    # Create a placeholder chart for popularity distribution
    popularity_chart = pn.panel(popularity_distribution())

    # Create a placeholder chart for key distribution
    key_distribution_chart = pn.panel(key_distribution())

    # Create a placeholder chart for time signature distribution
    time_signature_distribution_chart = pn.panel(time_signature_distribution())

    # Create a placeholder chart for explicit distribution (pie chart)
    explicit_distribution_chart = pn.panel(explicit_distribution())

    danceability_chart=pn.panel(danceability_box_plot())

    # Create a Panel plot for the initial attribute (popularity)
    attribute_average_chart = pn.panel(attribute_averages_by_genre(), sizing_mode="stretch_width")

    # Create a placeholder chart for popularity vs. energy
    popularity_energy_chart = pn.panel(popularity_vs_energy(), sizing_mode="stretch_both")


    # Call the function to create the correlation heatmap
    correlation_heatmap_pane = create_correlation_heatmap()







    # Add the callback to the genre_selector
    genre_selector.param.watch(update_charts, 'value')

    # Add the callback to the attribute_selector
    attribute_selector.param.watch(update_attribute_chart, 'value')

    # Add a callback to the track_selector to update the popularity vs. energy chart
    track_selector.param.watch(update_popularity_energy_chart, 'value')

    # Store the initial chart states
    initial_popularity_chart = popularity_chart.object
    initial_key_distribution_chart = key_distribution_chart.object
    initial_time_signature_distribution_chart = time_signature_distribution_chart.object
    initial_explicit_distribution_chart = explicit_distribution_chart.object
    initial_danceability_chart= danceability_chart.object
    # Store the initial state of the attribute chart
    initial_attribute_chart = attribute_average_chart.object

    initial_popularity_energy_chart=popularity_energy_chart.object



    # Create a refresh button and define a callback function
    def refresh_charts(event):
        # Reset the charts to their initial state
        popularity_chart.object = initial_popularity_chart
        key_distribution_chart.object = initial_key_distribution_chart
        time_signature_distribution_chart.object = initial_time_signature_distribution_chart
        explicit_distribution_chart.object = initial_explicit_distribution_chart
        danceability_chart.object = initial_danceability_chart

    refresh_button.on_click(refresh_charts)

    # Create a refresh button for the attribute chart
    def refresh_attribute_chart(event):
        # Reset the attribute chart to its initial state
        attribute_average_chart.object = initial_attribute_chart

    attribute_refresh_button.on_click(refresh_attribute_chart)


    sidebar = pn.Column(
        pn.Column(
            genre_selector,
            refresh_button,
            attribute_selector,
            attribute_refresh_button,
        )
    )

    app_layout = pn.Column(
        description,
        pn.layout.VSpacer(height=3),
        data_table,
        pn.layout.VSpacer(height=10),
        pn.Row(
            sidebar,
            pn.Column(
                pn.Tabs(
                    ('Popularity Distribution', popularity_chart),
                    ('Key Distribution', key_distribution_chart),
                    ('Time Signature Distribution', time_signature_distribution_chart),
                    ('Explicit Tracks Distribution', explicit_distribution_chart),
                    ('Danceability by Genre', danceability_chart),
                ),
                attribute_average_chart,
                correlation_heatmap_pane,
                sizing_mode="stretch_width",
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
