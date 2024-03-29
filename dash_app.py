from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.express as px
import numpy as np
import os
import pandas as pd
from pathlib import Path
import src.predictions_with_land_mask as preds


# Path to files
ais_file = 'data/ais_datalastic_filtered.csv'
timestamp_file = 'data/timestamps_sar_images.csv'

bucket_id = os.environ.get('SAR_BUCKET')
local_path = f'data/{bucket_id}/VH'
# local_path = 'data/bucket/VH'

# Get list of SAR images from local path
list_of_imgs = os.listdir(local_path)

# Coordinates for Laconian Bay
latitude = 36.53353  # 36.34289 
longitude = 22.721728  # 22.43289
basemap = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

# Create a base map
# fig.update_layout(mapbox_style="open-street-map")
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})


# Initialize the app
# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        dcc.DatePickerSingle(
            id='date-picker',
            date='2022-03-04',  # datetime.date.today()
            display_format='YYYY-MM-DD'
        ),
    ],
    style=SIDEBAR_STYLE,
)

fig = px.scatter_mapbox(basemap, lat='lat', lon='lon', zoom=10, mapbox_style='carto-darkmatter')  # open-street-map

content = html.Div([
    dcc.Graph(figure=fig, id='graph-content'),
    dcc.Dropdown(list_of_imgs, list_of_imgs[0], id='dropdown-selection'),
    # Add other components here
], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"), 
    sidebar, 
    content
])


# App layout

# app.layout = html.Div(children=[
#     dbc.Row(dbc.Col()),
#     dbc.Row(dbc.Col())
# ])
# app.layout = html.Div([
#     dbc.Row([
#         dbc.Col([
#             html.H2('Sidebar'),
#             dcc.DatePickerSingle(
#                 id='date-picker',
#                 date=datetime.date.today(),
#                 display_format='YYYY-MM-DD'
#             ),
#             # Add your sidebar content here
#         ], md=4),  # This sets the size of the sidebar column to 4 out of 12
#         dbc.Col([
#             html.H1(children='DarkVessel', style={'textAlign': 'center'}),
#             dcc.Dropdown(list_of_imgs, list_of_imgs[0], id='dropdown-selection'),
#             dcc.Graph(figure=fig, id='graph-content'),
#         ], md=8),  # This sets the size of the graph column to 8 out of 12 (Bootstrap grid system)
#     ])
# ])


# app.layout = html.Div([
#     html.H1(children='DarkVessel', style={'textAlign': 'center'}),
#     dcc.Dropdown(list_of_imgs, list_of_imgs[0], id='dropdown-selection'),
#     dcc.DatePickerSingle(
#         id='date-picker',
#         date=datetime.date.today(),
#         display_format='YYYY-MM-DD'
#     ),
#     dcc.Graph(figure=fig, id='graph-content'),
# ])


@callback(
    Output('graph-content', 'figure'), 
    Input('date-picker', 'date')
)
def update_graph(date):
    # Reading timestamp data from SAR images
    df_timestamp = pd.read_csv(timestamp_file)
    df_timestamp = df_timestamp.set_index('TILE_ID')
    
    # Get image ID from date picker value
    mask_date = df_timestamp['DATE'] == date
    image_list = list(df_timestamp[mask_date].index)
    # image_id = image_list[0]  # Select the first image for testing
    # Run prediction on selected image
    
    predictions = []
    
    # Run predictions on all images for the selected date
    for image_id in image_list:
        image_file = f'{image_id}.tif'
        print(f"Predictions on {image_file}")
        df_preds = preds.predict(image_file, plot=False)
        predictions.append(df_preds)
    # Concatenate predictions
    df_preds = pd.concat(predictions, ignore_index=True, axis=0)
    # df_preds = pd.read_csv('data/mask_test.csv')  # Read predictions from file
    
    # Reading AIS data from datalastic dataset
    df_ais = pd.read_csv(ais_file)
    
    # Get timestamp for selected image
    image_timestamp = df_timestamp.loc[image_list[0], 'TIMESTAMP']
    
    # Filter for date range of SAR image
    timedelta=1
    start = pd.to_datetime(image_timestamp) - pd.Timedelta(timedelta, 'hour')
    end = pd.to_datetime(image_timestamp) + pd.Timedelta(timedelta, 'hour')

    df_ais['timestamp'] = pd.to_datetime(df_ais['timestamp'])
    df_ais_filtered = df_ais[df_ais.timestamp.between(start, end)].copy()

    # Aggregate coordinates for each vessel
    df_ais_agg = df_ais_filtered.groupby(['name', 'mmsi']).agg({'lat': 'mean', 'lon': 'mean'}).reset_index()
    
    df_ais_agg['prediction'] = 0
    df_preds.columns = ['name', 'lat', 'lon', 'prediction']
    tbp = pd.concat([df_ais_agg, df_preds], axis=0, ignore_index=True)

    fig = px.scatter_mapbox(
        tbp,
        lat="lat",
        lon="lon",
        color="prediction",
        zoom=9.5,
        height=800,
        mapbox_style="carto-positron",
        hover_data=['name', 'lat', 'lon', 'prediction']
    )
    # fig.update_layout(mapbox_style="carto-positron")  # open-street-map, carto-darkmatter
    # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update(layout_coloraxis_showscale=False)
    # Update markers to circles and increase their size
    fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
    fig.update_traces(
        marker=dict(
            size=15, # Adjust the size as needed
            symbol="circle", # Set symbol to 'circle'
        ),
        selector=dict(mode="markers"),
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
