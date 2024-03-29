from dash import Dash, html, dcc, callback, Output, Input
from dash import dash_table
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.express as px
import numpy as np
import os
import pandas as pd
from pathlib import Path
import src.predictions_with_land_mask as preds



# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Timestamps for existing SAR images
timestamp_file = 'data/timestamps_sar_images.csv'
df_timestamp = pd.read_csv(timestamp_file)
df_timestamp = df_timestamp.set_index('TILE_ID')

# AIS data from datalastic dataset
ais_file = 'data/ais_datalastic_filtered.csv'
df_ais = pd.read_csv(ais_file)
df_ais['timestamp'] = pd.to_datetime(df_ais['timestamp'])

# Path to the SAR images
bucket_id = os.environ.get('SAR_BUCKET')
local_path = f'data/{bucket_id}/VH'
# local_path = 'data/bucket/VH'

# Get list of SAR images from local path
list_of_imgs = os.listdir(local_path)

# Base map for Laconian Bay
latitude = 36.53353  # 36.34289 
longitude = 22.721728  # 22.43289
fig = px.scatter_mapbox(
    lat=[latitude],
    lon=[longitude],
    zoom=10,
    height=800,
    mapbox_style='carto-positron',  # open-street-map
)
fig.update_traces(
    marker=dict(
        size=0, # Adjust the size as needed
        symbol="circle", # Set symbol to 'circle'
    ),
    selector=dict(mode="markers"),
)
fig.update_layout(mapbox_bounds={"west": 22.357, "east": 23.114, "south": 36.373, "north": 36.846})


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "display": "inline-block",
}

MAP_STYLE = {
    "display": "inline-block",
    "width": "60%",
    "padding": "2rem 1rem",
}

REPORT_STYLE = {
    "position": "fixed",
    "top": 0,
    "right": 0,
    "width": "20%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "display": "inline-block",
    "vertical-align": "top"
}

sidebar_content = html.Div([
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
])

map_content = html.Div([
    html.H1(children='DarkVessel', style={'textAlign': 'center'}),
    dcc.Graph(figure=fig, id='graph-content', clickData=None),
])

report_content = html.Div([
    dash_table.DataTable(id='click-output', columns=[
        {"name": i, "id": i} for i in ["Name", "Lat", "Lon"]
    ])
])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left sidebar
                dbc.Col(
                    sidebar_content,
                    width=2, # 2 out of 12 columns
                    style={"background-color": "#f8f9fa", "padding": "2rem 1rem"},
                ),
                # Map
                dbc.Col(
                    map_content,
                    width=8, # 8 out of 12 columns
                    style={"padding": "2rem 1rem"},
                ),
                # Right sidebar
                dbc.Col(
                    report_content,
                    width=2, # 2 out of 12 columns
                    style={"background-color": "#f8f9fa", "padding": "2rem 1rem"},
                ),
            ],
        )
    ],
    fluid=True,
)

# app.layout = html.Div([
#     dcc.Location(id="url"), 
#     sidebar, 
#     map,
#     report
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
    # Get image ID from date picker value
    mask_date = df_timestamp['DATE'] == date
    image_list = list(df_timestamp[mask_date].index)
    
    predictions = []
    
    # Run predictions on all images for the selected date
    # for image_id in image_list:
    #     image_file = f'{image_id}.tif'
    #     print(f"Predictions on {image_file}")
    #     df_preds = preds.predict(image_file, plot=False)
    #     predictions.append(df_preds)
    # # Concatenate predictions
    # df_preds = pd.concat(predictions, ignore_index=True, axis=0)
    
    df_preds = pd.read_csv('data/mask_test.csv')  # Read predictions from file
    df_preds.columns = ['name', 'lat', 'lon', 'prediction']
    
    # Get timestamp for selected image
    image_timestamp = df_timestamp.loc[image_list[0], 'TIMESTAMP']
    
    # Filter for date range of SAR image
    timedelta = 45  # minutes
    start = pd.to_datetime(image_timestamp) - pd.Timedelta(timedelta, 'min')
    end = pd.to_datetime(image_timestamp) + pd.Timedelta(timedelta, 'min')
    df_ais_filtered = df_ais[df_ais.timestamp.between(start, end)].copy()

    # Aggregate coordinates for each vessel
    df_ais_agg = df_ais_filtered.groupby(['name', 'mmsi']).agg({'lat': 'mean', 'lon': 'mean'}).reset_index()
    df_ais_agg['prediction'] = 0
    
    tbp = pd.concat([df_ais_agg, df_preds], axis=0, ignore_index=True)

    fig = px.scatter_mapbox(
        tbp,
        lat="lat",
        lon="lon",
        color="prediction",
        zoom=8,
        height=800,
        mapbox_style="carto-positron",
        hover_data=['name', 'lat', 'lon', 'prediction']
    )
    # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update(layout_coloraxis_showscale=False)
    # fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
    fig.update_layout(mapbox_bounds={"west": 22.357, "east": 23.114, "south": 36.373, "north": 36.846})
    
    fig.update_traces(
        marker=dict(
            size=15, # Adjust the size as needed
            symbol="circle", # Set symbol to 'circle'
        ),
        selector=dict(mode="markers"),
    )
    
    return fig

@app.callback(
    Output('click-output', 'data'),
    Input('graph-content', 'clickData'))
def display_click_data(clickData):
    if clickData is None:
        return [{}]
    else:
        point_data = clickData['points'][0]
        return [{
            "Name": point_data['customdata'][0],
            "Lat": point_data['lat'],
            "Lon": point_data['lon']
        }]



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
