# Spotiviz - Spotify Tracks Data Application

## Overview

Spotiviz is a data application that allows users to explore, analyze, and predict Spotify tracks' data through interactive dashboards. The application is built using [FastAPI](https://fastapi.tiangolo.com/), [Panel](https://panel.holoviz.org/), and [Bokeh](https://docs.bokeh.org/). It provides three dashboards, each offering a unique perspective on the Spotify tracks dataset.

## Installation

To run Spotiviz, follow these steps:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

2. Install the required packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application using [uvicorn](https://www.uvicorn.org/):
   ```bash
   uvicorn main:app --reload
   ```

4. Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser to access the home page.

## Dashboards

Spotiviz offers the following dashboards:

1. **Home Page**: Navigate to the home page to get an overview of the application.

2. **Dashboard 1 - Exploratory Dashboard**: Explore the Spotify tracks dataset with various filters and visualizations. Uncover patterns, trends, and details of top songs, artists, and genres.

    - Access: [http://127.0.0.1:8000/firstdash](http://127.0.0.1:8000/firstdash)

3. **Dashboard 2 - Genre Prediction Dashboard**: Predict and understand the genre of a track using machine learning models. Interact with predictions and discover the magic behind the music.

    - Access: [http://127.0.0.1:8000/seconddash](http://127.0.0.1:8000/seconddash)

4. **Dashboard 3 - Further Analysis**: For those who crave more, this dashboard provides a space for additional insights, allowing you to explore fascinating aspects of the dataset that go beyond genre prediction.

    - Access: [http://127.0.0.1:8000/thirddash](http://127.0.0.1:8000/thirddash)

## Meet the Team

Spotiviz is brought to you by our enthusiastic duo:

- [**Ines Ouhichi**](https://www.linkedin.com/in/ines-ouhichi/)

- [**Selima Gadri**](https://www.linkedin.com/in/selima-gadri/)

## Contact

For inquiries or support, feel free to reach out to us:

- Selima Gadri: [selima.gadri@dauphine.tn](mailto:selima.gadri@dauphine.tn)
- Ines Ouhichi: [ines.ouhichi@dauphine.tn](mailto:ines.ouhichi@dauphine.tn)

## Additional Information

- GitHub Repository: [https://github.com/selimagadri/dataviz_project](https://github.com/selimagadri/dataviz_project)
- Dataset Source: [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

Explore the world of music on Spotify with Spotiviz! 🎶
