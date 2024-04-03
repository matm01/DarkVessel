from dash import Dash, html, dcc, callback, Output, Input, State
from dash import dash_table, no_update, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
import src.predictions_with_land_mask as preds



# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
load_figure_template('SPACELAB')

# Timestamps for existing SAR images
timestamp_file = 'data/timestamps_sar_images.csv'
df_timestamp = pd.read_csv(timestamp_file)
df_timestamp = df_timestamp.set_index('TILE_ID')
sar_dates = df_timestamp['DATE'].unique().tolist()

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

# Mabox token
token = os.environ.get('MAPBOX_TOKEN')

# Base map for Laconian Bay
latitude = 36.53353  # 36.34289 
longitude = 22.721728  # 22.43289


#==============================================================================
# Define the content of the app
#==============================================================================


header_content = html.Div([
    html.H1('My Dashboard', id='header'),
])

#==============================================================================

sidebar_content = html.Div([
    html.H2("Sidebar", className="display-4"),
    html.Hr(),
    html.P(
        "A simple sidebar layout with navigation links", className="lead"
    ),
    dbc.Nav(
        [
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("About", href="/about", active="exact"),
            # dbc.NavLink("Page 2", href="/page-2", active="exact"),
        ],
        vertical=True,
        pills=True,
    ),
    # dcc.Dropdown(
    #     id='date-dropdown',
    #     options=[{'label': date, 'value': date} for date in sar_dates],
    #     value=sar_dates[0] if sar_dates else None,  # Sets the default value to the first date
    # ),
    # dbc.Button('Previous', id='prev-btn', n_clicks=0),
    # dbc.Button('Next', id='next-btn', n_clicks=0),
    # html.Div(id='selected-date-display'),
    # daq.BooleanSwitch(id='boolean-switch', on=False),
    # html.Div(id='boolean-switch-output-1')
    
])

#==============================================================================

controls = dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='date-dropdown',
                options=[{'label': date, 'value': date} for date in sar_dates],
                value=sar_dates[0] if sar_dates else None, # Sets the default value to the first date
            ), width=2),
            dbc.Col(dbc.Button('Previous', id='prev-btn', n_clicks=0), width=1),
            dbc.Col(dbc.Button('Next', id='next-btn', n_clicks=0), width=1),
            # dbc.Col(html.Div(id='selected-date-display'), width=1),
            dbc.Col(daq.BooleanSwitch(id='boolean-switch', on=False), width=3),
            dbc.Col(html.Div(id='boolean-switch-output-1'), width=1),
            dbc.Col(dbc.Button('Run', id='run-button', n_clicks=0), width=2, align='end'),
])

#==============================================================================

map_content =  dbc.Row([
        dbc.Col([
            # dbc.Button('Run', id='run-button', n_clicks=0),
            dcc.Graph(
                id='base-map',
                config={'displayModeBar': False},
                clickData=None
            ),
            dash_table.DataTable(
                id='data-table',
                columns=[
                    {'name': 'Name', 'id': 'name'},
                    {'name': 'Latitude', 'id': 'lat'},
                    {'name': 'Longitude', 'id': 'lon'},
                    {'name': 'Prediction', 'id': 'prediction'}
                ],
                data=[]
            )
        ])
])

# map_content = html.Div([
#     dbc.Button('Run', id='run-button', n_clicks=0),
#     dcc.Graph(
#         id='base-map',
#         config={'displayModeBar': False},
#         clickData=None),
#     dash_table.DataTable(
#         id='data-table',
#         columns=[
#             {'name': 'Name', 'id': 'name'},
#             {'name': 'Latitude', 'id': 'lat'},
#             {'name': 'Longitude', 'id': 'lon'},
#             {'name': 'Prediction', 'id': 'prediction'}
#         ],
#         data=[]),
# ])

#==============================================================================

report_content = dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='click-output-data',
                columns=[
                    {"name": "Attribute", "id": "Attribute"}, 
                    {"name": "Value", "id": "Value"}
                ],
                style_cell_conditional=[
                    {'if': {'column_id': 'Attribute'}, 'width': '60px'},
                    {'if': {'column_id': 'Value'}, 'width': '240px'}
                ],
                style_cell={
                    'textAlign': 'left', 
                    'minWidth': '60px',
                    'maxWidth': '240px',
                    'whiteSpace': 'normal'
                },
                style_data=dict(height='20px'),
                style_table={'overflowX': 'auto'},
            ),
            html.Img(
                id='image-placeholder', 
                alt='Click on data point to display image',
                style={'width': '300px', 'height': '300px'}
            ),
        ], style={'top': '5rem'})
])

#==============================================================================
# Define the layout of the app
#==============================================================================

app.layout = dbc.Container([
        dbc.Row([
            # Header
            dbc.Col(
                header_content,
                style={'textAlign': 'left', "background-color": "#f8f9fa", "padding": "1rem 1rem"},
                width=12
            )
        ]),
        dbc.Row([
            # Left sidebar
            dbc.Col(
                sidebar_content,
                width=2, # 2 out of 12 columns
                style={"background-color": "#f8f9fa", "padding": "1rem 1rem"},
            ),
            # Map
            dbc.Col([
                controls,
                map_content,
            ], width=8, style={'background-color': '#f8f9fa', "padding": "1rem 1rem"}),
            # Right sidebar
            dbc.Col([
                report_content,
            ], width=2, style={"background-color": "#f8f9fa", "top": "2rem", "padding": "2rem 0rem"}),
        ])
    ], fluid=True,
)

#==============================================================================
# Define the app callbacks
#==============================================================================

# Update dates
@app.callback(
    Output('date-dropdown', 'value'),
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('date-dropdown', 'value')]
)
def update_dropdown(prev_clicks, next_clicks, current_value):
    ctx = callback_context
    if not ctx.triggered or not sar_dates:
        return sar_dates[0] if sar_dates else None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_index = sar_dates.index(current_value)
        if button_id == 'next-btn':
            new_index = min(len(sar_dates) - 1, current_index + 1)
        elif button_id == 'prev-btn':
            new_index = max(0, current_index - 1)
        else:
            new_index = current_index
        return sar_dates[new_index]

# Callback to display the selected date
# @app.callback(
#     Output('selected-date-display', 'children'),
#     [Input('date-dropdown', 'value')]
# )
# def display_selected_date(selected_date):
#     return f'Selected Date: {selected_date}'

#==============================================================================

# @app.callback(
#     Output('run-button', 'disabled'),
#     Input('date-dropdown', 'value'),
#     State('run-button', 'disabled')
# )
# def update_button_state(date, is_disabled):
#     # If a date is selected, enable the button
#     if date:
#         return False
#     # If no date is selected, keep the button disabled
#     return True


#==============================================================================

# Run the model on selected date
@app.callback(
    [Output('data-table', 'data'), Output('run-button', 'n_clicks')],
    # Output('data-table', 'data'),
    Input('run-button', 'n_clicks'),
    State('date-dropdown', 'value')
)
def run_model(n_clicks, date):
    if n_clicks > 0:
        # Get image ID from date picker value
        mask_date = df_timestamp['DATE'] == date
        image_list = list(df_timestamp[mask_date].index)
        
        # Get timestamp for selected image
        image_timestamp = df_timestamp.loc[image_list[0], 'TIMESTAMP']
        
        # For testing purposes, only use the first two images
        results_files = os.listdir('results')[:2]
        df_preds = pd.concat([pd.read_csv(f'results/{results_file}') for results_file in results_files], ignore_index=True, axis=0)
        
        # predictions = []
        # # Run predictions on all images for the selected date
        # for image_id in image_list:
        #     image_file = f'{image_id}.tif'
        #     print(f"Predictions on {image_file}")
        #     df_preds = preds.predict(image_file, plot=False)
        #     predictions.append(df_preds)
        # # Concatenate predictions
        # df_preds = pd.concat(predictions, ignore_index=True, axis=0)
        
        df_preds.columns = ['name', 'lat', 'lon', 'prediction', 'image']
        data = df_preds.to_dict('records')
        # print(f'Data table: {data}')
        return data, 1
    else:
        print('No data returned from run_model!')
        return [], 0

#==============================================================================

# Update map with predictions
@app.callback(
    Output('base-map', 'figure'),
    [Input('run-button', 'n_clicks'), Input('data-table', 'data')]
)
def update_map(n_clicks, data):
    if n_clicks > 0:
        print(f'Updating map on clicks with {data}')
        # # Convert data to DataFrame for easier manipulation
        fig = px.scatter_mapbox(
            data,
            lat="lat",
            lon="lon",
            color="prediction",
            zoom=9,
            height=700,
            mapbox_style="carto-positron",
            hover_data=['name', 'lat', 'lon', 'prediction', 'image']
        )
        fig.update_layout(
            # mapbox_bounds={"west": 22.35, "east": 23.12, "south": 36.35, "north": 36.85},
            margin={"r": 5, "t": 5, "l": 5, "b": 5},
            # mapbox_style="streets",
            mapbox_style="satellite-streets",
            # mapbox_style="mapbox://styles/mapbox/navigation-guidance-night-v2",
            mapbox_accesstoken=token
        )
        fig.update(layout_coloraxis_showscale=False)
        fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
        return fig
    # else:
    print(f'Update map no clicks: {data}')
    fig = px.scatter_mapbox(
        lat=[latitude],
        lon=[longitude],
        zoom=9,
        height=700,
        mapbox_style='carto-positron',  # open-street-map
    )
    fig.update_traces(
        marker=dict(
            size=0, # Adjust the size as needed
            symbol="circle", # Set symbol to 'circle'
        ),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        # mapbox_bounds={"west": 22.35, "east": 23.12, "south": 36.35, "north": 36.85},
        margin={"r": 5, "t": 5, "l": 5, "b": 5},
        mapbox_style="mapbox://styles/mapbox/navigation-guidance-night-v2",
        mapbox_accesstoken=token
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
    return fig  

#==============================================================================

# Define a callback to show AIS data when toggle switch is on
@app.callback(
    Output('base-map', 'figure', allow_duplicate=True),
    Input('boolean-switch', 'on'),
    [State('date-dropdown', 'value'), State('base-map', 'figure')],
    prevent_initial_call=True
)
def update_map(toggle_on, date, current_figure):
     # Get image ID from date picker value
    mask_date = df_timestamp['DATE'] == date
    image_list = list(df_timestamp[mask_date].index)
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
    if toggle_on:
        # Show all data points
        fig = px.scatter_mapbox(
            df_ais_agg,
            lat="lat",
            lon="lon",
            color="prediction",
            zoom=8,
            height=700,
            mapbox_style="carto-positron",
            hover_data=['name', 'mmsi','lat', 'lon']
        )
        # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update(layout_coloraxis_showscale=False)
        # fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
        # fig.update_layout(mapbox_bounds={"west": 20.0, "east": 25.0, "south": 35.0, "north": 37.0})
        
        fig.update_traces(
            marker=dict(
                size=15, # Adjust the size as needed
                symbol="circle", # Set symbol to 'circle'
            ),
            selector=dict(mode="markers"),
        )
        return fig
    else:
        return current_figure

# @app.callback(
#     Output('base-map', 'figure'), 
#     [Input('toggle-switch', 'on')],
#     [State('date-picker', 'date'), State('base-map', 'figure')],
# )
# def update_graph(toggle_on, date, current_figure):
#     if toggle_on:
#         # Get image ID from date picker value
#         mask_date = df_timestamp['DATE'] == date
#         image_list = list(df_timestamp[mask_date].index)
#         # Get timestamp for selected image
#         image_timestamp = df_timestamp.loc[image_list[0], 'TIMESTAMP']
        
#         # Filter for date range of SAR image
#         timedelta = 45  # minutes
#         start = pd.to_datetime(image_timestamp) - pd.Timedelta(timedelta, 'min')
#         end = pd.to_datetime(image_timestamp) + pd.Timedelta(timedelta, 'min')
#         df_ais_filtered = df_ais[df_ais.timestamp.between(start, end)].copy()

#         # Aggregate coordinates for each vessel
#         df_ais_agg = df_ais_filtered.groupby(['name', 'mmsi']).agg({'lat': 'mean', 'lon': 'mean'}).reset_index()
#         df_ais_agg['prediction'] = 0
        
#         fig = px.scatter_mapbox(
#             df_ais_agg,
#             lat="lat",
#             lon="lon",
#             color="prediction",
#             zoom=8,
#             height=700,
#             mapbox_style="carto-positron",
#             hover_data=['name', 'mmsi','lat', 'lon']
#         )
#         # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
#         fig.update(layout_coloraxis_showscale=False)
#         # fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
#         # fig.update_layout(mapbox_bounds={"west": 20.0, "east": 25.0, "south": 35.0, "north": 37.0})
        
#         fig.update_traces(
#             marker=dict(
#                 size=15, # Adjust the size as needed
#                 symbol="circle", # Set symbol to 'circle'
#             ),
#             selector=dict(mode="markers"),
#         )
#         return fig
#     else:
#         # The switch is in the "off" state
#         # Update the graph with a different set of data
#         return current_figure

#==============================================================================

# @app.callback(
#     Output('click-output', 'data'),
#     Input('base-map', 'clickData'))
# def display_click_data(clickData):
#     if clickData is None:
#         return [{}]
#     else:
#         point_data = clickData['points'][0]
#         return [{
#             "Name": point_data['customdata'][0],
#             "Lat": point_data['lat'],
#             "Lon": point_data['lon']
#         }]

def decimal_to_dms_latitude(decimal_lat):
    is_negative = decimal_lat < 0
    decimal_lat = abs(decimal_lat)
    degrees = int(decimal_lat)
    minutes = int((decimal_lat - degrees) * 60)
    seconds = (decimal_lat - degrees - minutes/60) * 3600
    direction = "S" if is_negative else "N"
    return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"

def decimal_to_dms_longitude(decimal_lon):
    is_negative = decimal_lon < 0
    decimal_lon = abs(decimal_lon)
    degrees = int(decimal_lon)
    minutes = int((decimal_lon - degrees) * 60)
    seconds = (decimal_lon - degrees - minutes/60) * 3600
    direction = "W" if is_negative else "E"
    return f"{degrees}°{minutes}'{seconds:.1f}\"{direction}"


@app.callback(
    Output('click-output-data', 'data'),
    Input('base-map', 'clickData'))
def display_click_data_table(clickData):
    if clickData is None:
        return [{}]
    else:
        point_data = clickData['points'][0]
        lat_formatted = decimal_to_dms_latitude(point_data.get('lat'))
        lon_formatted = decimal_to_dms_longitude(point_data.get('lon'))
        data = [
            # {"Attribute": "Name", "Value": point_data.get('name')},
            {"Attribute": "Name", "Value": point_data.get('customdata', [None])[0]},
            {"Attribute": "Latitude", "Value": lat_formatted},
            {"Attribute": "Longitude", "Value": lon_formatted},
            {"Attribute": "Prediction", "Value": point_data.get('prediction')},
            # {"Attribute": "Name", "Value": point_data.get('customdata', [None])[4]},
            # {"Attribute": "Image", "Value": point_data.get('image')}
        ]
    return data


@app.callback(
    Output('image-placeholder', 'src'),
    Input('base-map', 'clickData'))
def display_click_data_image(clickData):
    if clickData is None:
        return ''
    else:
        try:
            point_data = clickData['points'][0]
            image = point_data.get('customdata', 'no image')[4]  # local path to image
            return image
        except IndexError:
            return ''


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
