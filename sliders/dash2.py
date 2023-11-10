import pandas as pd
import numpy as np
import panel as pn
from sklearn.cluster import KMeans
import holoviews as hv
from holoviews import opts
import hvplot.pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import bokeh.plotting as bpl
from bokeh.models import ColorBar, LinearColorMapper, ColumnDataSource
from bokeh.palettes import Greens



def createClassification():
    hv.extension('bokeh')

    # Read dataset
    data = pd.read_csv('dataset.csv', 
                        index_col=0
                       )
    # SAMPLED DATA
    # #Create an empty DataFrame to store the sampled data
    # data = pd.DataFrame()
    # # Group the original DataFrame by 'track_genre'
    # grouped = df1.groupby('track_genre')
    # # Iterate over each group, and sample 20% of the data for each 'track_genre'
    # for genre, group_df in grouped:
    #     data = pd.concat([data, group_df.sample(frac=0.1, random_state=42)])
    # # Reset the index of the sampled data
    # data.reset_index(drop=True, inplace=True)

    # Drop duplicates and naan
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    #Data preprocessing
    data['explicit'] = data['explicit'].replace({True: 1, False: 0})
    label_encoder = LabelEncoder()
    data['track_genre'] = label_encoder.fit_transform(data['track_genre'])

    X_before = data.drop(columns=['track_genre','track_id','artists','album_name','track_name']) 
    # Initialize the MinMaxScaler and Fit and transform the data
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X_before), columns=X_before.columns) # Features
    y = data['track_genre']  # Target variable

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define classifier options
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': '',
        'XGBoost': XGBClassifier(),
        'CatBoost':  CatBoostClassifier()
    }
    # Clssifier selector
    classifier_selector = pn.widgets.Select(options=list(classifiers.keys()), value='Naive Bayes', name='Select Model')

    # Train and evaluate selected model 
    @pn.depends(classifier_selector.param.value)
    def train_and_evaluate(model_name):
        if model_name == 'Neural Network':
            
            print(model_name, ': start training')
           
            X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
            X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
            y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
            y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)
            y_train_one_hot = to_categorical(y_train_tf)

            model = Sequential()
            model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(114, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

            model.fit(X_train_tf, y_train_one_hot, epochs=10, batch_size=64, validation_split=0.1)

            y_pred_prob = model.predict(X_test_tf)
            y_pred = y_pred_prob.argmax(axis=1)


        else:
            classifier = classifiers[model_name]
            print(model_name, ': start training')

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)

        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
        class_report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        report = pd.DataFrame(class_report).transpose()
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")

        #confusion = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        confusion = conf_matrix
        confusion_matrix_df = pd.DataFrame(confusion)
        confusion_matrix_df.columns = label_encoder.inverse_transform(confusion_matrix_df.columns.astype(int))
        confusion_matrix_df.index = label_encoder.inverse_transform(confusion_matrix_df.index.astype(int))
        print(confusion_matrix_df.head())
        print(model_name, ': end training')

        return report, accuracy, confusion_matrix_df
    


    def create_confusion_matrix_heatmap(confusion_matrix_df):
        mat = confusion_matrix_df

        custom_palette = list(Greens[9])
    
        color_mapper = LinearColorMapper(palette=custom_palette, low=mat.min().min(), high=mat.max().max())

        flat_values = mat.values.flatten()

        x_vals = [col for col in mat.columns for _ in range(len(mat))]
        y_vals = [index for _ in range(len(mat.columns)) for index in mat.index]
        color_vals = flat_values

        source = ColumnDataSource(data=dict(
            x=x_vals,
            y=y_vals,
            image=color_vals,
        ))

        heatmap_figure = bpl.figure(
            title="Confusion Matrix",
            x_range=[str(col) for col in mat.columns],
            y_range=[str(index) for index in mat.index],
            width=700,
            height=700,
            tools="hover",
            tooltips=[('True Class', '@y'), ('Predicted Class', '@x'), ('Value', '@image')],
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

        heatmap_figure.xaxis.major_label_orientation = np.pi / 2
        heatmap_figure.xaxis.major_label_text_font_size = "5pt"
        heatmap_figure.yaxis.major_label_text_font_size = "5pt"

        heatmap_pane = pn.pane.Bokeh(heatmap_figure, 
                                     #sizing_mode="stretch_both"
                                     )

        return heatmap_pane

    
    @pn.depends(classifier_selector.param.value)
    def update_dashboard(model_name):
        report, accuracy, confusion_matrix_df = train_and_evaluate(model_name)
        number = pn.indicators.Number(
            name='Accuracy', value=72, format='{value}%',
            colors=[(33, 'red'), (66, 'gold'), (100, 'green')]
            )

        classification_report_panel = pn.Column(
            pn.pane.HTML("<h2 style='color: green;'>Classification Report</h2>", width=800),
            pn.widgets.DataFrame(report)
            )
            
        return pn.Row(
            pn.Column(
                number.clone(value=round(accuracy*100,2)),
                classification_report_panel,
                #styled_df_widget,
                
            ),
            pn.Column(create_confusion_matrix_heatmap(confusion_matrix_df), 
                      #sizing_mode='fixed'
                      )
        )

    dashboard = pn.Column(
        pn.Column(
                pn.pane.Markdown("""
                # Spotify Track Genre Prediction
                
                This dashboard provides an overview of the Spotify Track Genre Prediction model.
                
                - **Accuracy**: Displays the accuracy of the model.
                - **Classification Report**: Shows detailed metrics on the model's performance.
                - **Confusion Matrix Heatmap**: Visual representation of the model's confusion matrix.
                
                Use the classifier selector to choose different models.
                """
                                 )
                                 ),
        classifier_selector,
        update_dashboard
    )

    return dashboard


def createClustering():
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

def createApp2():
    classification_section = createClassification()
    clustering_section = createClustering()

    tabs = pn.Tabs(("Classification", classification_section), 
                   ("Clustering", clustering_section)
                   )

    return pn.Column(
        pn.pane.Markdown("# Spotify Track Analysis"),
        tabs
    ).servable()

        