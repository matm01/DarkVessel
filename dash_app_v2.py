from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from geopy import distance
import numpy as np
import os
import pandas as pd
import plotly.express as px


app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
load_figure_template('LUX')


# Timestamps for existing SAR images
timestamp_file = 'data/timestamps_sar_images.csv'
df_timestamp = pd.read_csv(timestamp_file)
df_timestamp = df_timestamp.set_index('TILE_ID')
unique_dates = df_timestamp['DATE'].unique()
list_of_unique_dates = unique_dates.tolist()

# Results from the predictions
results_file = 'results/results_all.csv'
df_results = pd.read_csv(results_file)

# Map
token = os.environ.get('MAPBOX_TOKEN')
latitude = 36.53353  # 36.34289 
longitude = 22.721728  # 22.43289

#==============================================================================
# Define the style of the app
#==============================================================================

BACKGROUND_COLOR = "#f8f9fa"

#==============================================================================
# Define the content of the app
#==============================================================================

header = dbc.Row([
    dbc.Col([
        html.H1("Ship-to-Ship transfer detection in SAR images", id='header', className="mb-4")
    ])
])

sidebar = dbc.Row([
    dbc.Col([
        # html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("Select start date", className="lead"),
        dcc.Dropdown(
            id='start-date',
            options=[{'label': date, 'value': date} for date in list_of_unique_dates],
            value=list_of_unique_dates[9] if list_of_unique_dates else None, # Sets the default value to the first date
        ),
        html.Br(),
        html.P("Select end date", className="lead"),
        dcc.Dropdown(
            id='end-date',
            options=[{'label': date, 'value': date} for date in list_of_unique_dates],
            value=list_of_unique_dates[-1] if list_of_unique_dates else None, # Sets the default value to the first date
        ),
        html.Br(),
        dbc.Button("Filter", id='filter-button', n_clicks=0),
        html.Br(),
        html.Hr(),
        html.H3("Summary statistics"),  # , className="lead"
        dash_table.DataTable(
            id='summary-table',
            columns=[
                {"name": "Description", "id": "description"}, 
                {"name": "", "id": "statistics"}
            ],
            style_cell_conditional=[
                {'if': {'column_id': 'Description'}, 'width': '250px', 'textAlign': 'right'},
                {'if': {'column_id': 'Statistics'}, 'width': '50px', 'textAlign': 'center',}
            ],
            style_cell={
                'textAlign': 'left', 
                'minWidth': '50px',
                'maxWidth': '250px',
                'whiteSpace': 'normal',
            },
            style_data={
                'height':'20px',
                'border': 'none',
                },
            style_table={'overflowX': 'auto'},
            style_header={'display': 'none'}
        ),        
        dcc.Store(id='data-table'),
    ])
])

control = dbc.Row([
    dbc.Col(dbc.ButtonGroup([
        dbc.Button('Previous', color='secondary', id='prev-btn', n_clicks=0),
        dbc.Button('Next', color='secondary', id='next-btn', n_clicks=0),
    ]), style={'width': 3, 'align':'end'}),
    dbc.Col(html.H2(id='frame-date'), width=3),
])


interactive_map = dbc.Row([
    dbc.Col([
        dcc.Graph(
            id='base-map',
            config={'displayModeBar': False},
            clickData=None
        ),
        html.Br() 
    ])
])


sar_ais_match = dbc.Row([
        dbc.Col([
            html.H3("Matching detections with AIS data"),
            dash_table.DataTable(
                id='sar-ais-match-table',
            ),
        ], style={'top': '5rem'})
])


ship_report = dbc.Row([
        dbc.Col([
            html.Hr(),
            html.H3("Vessel details"),  # , className="lead"
            dash_table.DataTable(
                id='click-output-data',
                columns=[
                    {"name": "Feature", "id": "Feature"}, 
                    {"name": "", "id": "Value"}
                ],
                style_cell_conditional=[
                    {'if': {'column_id': 'Feature'}, 'width': '70px'},
                    {'if': {'column_id': 'Value'}, 'width': '230px'}
                ],
                style_cell={
                    'textAlign': 'left', 
                    'minWidth': '70px',
                    'maxWidth': '230px',
                    'whiteSpace': 'normal'
                },
                style_data=dict(height='20px', border='none'),
                style_table={'overflowX': 'auto'},
                style_header={'display': 'none'}
            ),
            html.Img(
                id='image-placeholder', 
                alt='Click on data point to display image',
                style={'width': '410px', 'height': '375px'}
            ),
        ], style={'top': '5rem'})
])

#==============================================================================
# Define the layout of the app
#==============================================================================

app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(
                header,
                width=12,
                style={'textAlign': 'left',  "padding": "1rem 2rem"}
            ),
        ]),
        # Sidebar
        dbc.Row([
            dbc.Col(
                sidebar,
                width=2, style={'textAlign': 'left',"padding": "3rem 1rem"},
            ),
            dbc.Col([
                control,
                interactive_map,
                sar_ais_match,
                ],width=7, style={"padding": "1rem 1rem"}
            ),
            dbc.Col([
                ship_report,
                ], width=3, style={"padding": "3rem 1rem"}),
        ]),
    ], fluid=True
)

#==============================================================================
# Define the app callbacks
#==============================================================================

# Update end date options based on start date
@app.callback(
    [Output('end-date', 'options'),
    Output('frame-date', 'children', allow_duplicate=True)],
    Input('start-date', 'value'),
    prevent_initial_call=True
)
def update_end_date_options_and_frame_date(start_date):
    if start_date:
        # Ensure the end date is not before the start date
        end_dates = [{'label': date, 'value': date} for date in list_of_unique_dates if date > start_date]
        return end_dates, start_date
    else:
        # If no start date is selected, allow any date
        end_dates = [{'label': date, 'value': date} for date in list_of_unique_dates]
        return end_dates, start_date

#==============================================================================

# Filter results based on date range
@app.callback(
    Output('data-table', 'data'),
    Input('filter-button', 'n_clicks'),
    Input('start-date', 'value'),
    State('end-date', 'value')
)
def update_data_table(n_clicks, start_date, end_date):
    if start_date and end_date:
        mask_dates = (df_results['date'] >= start_date) & (df_results['date'] <= end_date)
        filtered_df = df_results[mask_dates]
        filtered_df = filtered_df.sort_values('date')
        data = filtered_df.to_dict('records')
        # print(data)
        return data
    else:
        data = df_results.to_dict('records')
        return None

#==============================================================================

# Filter results based on date range
@app.callback(
    Output('summary-table', 'data'),
    Input('filter-button', 'n_clicks'),
    State('data-table', 'data'),
    State('start-date', 'value'),
    State('end-date', 'value')
)
def update_summary_table(n_clicks, data, start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if n_clicks and data:
        df_filtered = pd.DataFrame(data)
        sts_detected =  df_filtered['prediction'].value_counts().loc['STS']
        ship_detected =  df_filtered['prediction'].value_counts().loc['Ship']
        number_of_days = (end - start)/ np.timedelta64(1, 'D')
        available_images = df_filtered['date'].nunique()
        revisit_frequency = 1 / (available_images / number_of_days)
        stats = [
            {"description": "STS detected", "statistics": sts_detected},
            {"description": "Ship detected", "statistics": ship_detected},
            {"description": "Period (days)", "statistics": number_of_days},
            {"description": "SAR images", "statistics": available_images},
            {"description": "Revisit frequency (days)", "statistics": round(revisit_frequency, 2)},
            
        ]
        return stats  #, start_date
    else:
        data = [{}]
        return data  #, start_date
#==============================================================================

# Update date for map based on frame selection using previous and next buttons
@app.callback(
    Output('frame-date', 'children'),
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks'),
    Input('start-date', 'value'),],
    [State('end-date', 'value'),
    State('frame-date', 'children')]
)
def update_frame_date(prev_clicks, next_clicks, start_date, end_date, frame_date):
    mask_dates = (unique_dates >= start_date) & (unique_dates <= end_date)
    filtered_dates = unique_dates[mask_dates].tolist()
    ctx = callback_context
    if not ctx.triggered or not filtered_dates:
        return filtered_dates[0] if filtered_dates else None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_index = filtered_dates.index(frame_date)
        if button_id == 'next-btn':
            new_index = min(len(filtered_dates) - 1, current_index + 1)
        elif button_id == 'prev-btn':
            new_index = max(0, current_index - 1)
        else:
            new_index = current_index
        return filtered_dates[new_index]

#==============================================================================

# Update map with predictions
@app.callback(
    Output('base-map', 'figure'),
    Input('frame-date', 'children'),
    Input('data-table', 'data')
)
def update_map(frame_date, data):
    df_results = pd.DataFrame(data)
    mask_date = df_results['date'] == frame_date
    df_results = df_results[mask_date]
    if data:
        # print(f'Updating map on clicks with {data}')
        # # Convert data to DataFrame for easier manipulation
        fig = px.scatter_mapbox(
            df_results,
            lat="latitude",
            lon="longitude",
            color="prediction",
            zoom=10,
            height=700,
            hover_data=[
                'latitude', 'longitude', 'mmsi', 'name', 'country', 
                'timestamp', 'timedelta', 'prediction', 'image', 'date'
                ],
            labels={
                "Ship": "Ship",
                "STS": "STS",
                "Not available (AIS)": "AIS",
                 },
            color_discrete_map={
                "STS": "red",
                "Ship": "green",
                "Not available (AIS)": "yellow"
            },
        )
        fig.update_traces(
            marker=dict(size=12, symbol="circle"),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            hovermode='closest',
            margin={"r": 10, "t": 10, "l": 10, "b": 10},
            # mapbox_style="carto-positron",
            # mapbox_style="streets",
            mapbox_style="satellite-streets",
            # mapbox_style="mapbox://styles/mapbox/navigation-guidance-night-v2",
            mapbox_accesstoken=token,
            legend=dict(
                font_size=15,
                font_color='white',
                title=dict(text=''),  # No title
                orientation="h",  # Horizontal orientation
                yanchor="top",  # Anchor legend at the bottom
                y=1.05,  # Position the legend below the plot
                xanchor="center",  # Center the legend horizontally
                x=0.5  # Center position of the legend (0.5 is the middle)
            ),
            legend_bgcolor='rgba(26, 26, 26, 1)',
            paper_bgcolor='rgba(26, 26, 26, 1)',
            
        )
        fig.update(layout_coloraxis_showscale=False)
        fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
        return fig
    
    fig = px.scatter_mapbox(
        lat=[latitude],
        lon=[longitude],
        zoom=8,
        height=700,
    )
    fig.update_traces(
        marker=dict(size=0, symbol="circle"),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        margin={"r": 5, "t": 5, "l": 5, "b": 5},
        mapbox_style="mapbox://styles/mapbox/navigation-guidance-night-v2",
        mapbox_accesstoken=token,
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.5,  # Position the legend below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.5  # Center position of the legend (0.5 is the middle)
        ),
        # paper_bgcolor='lightskyblue',
        # plot_bgcolor='lightblue' 
        # plot_bgcolor='rgba(100,100,100,1)'
        
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_mapboxes(center=dict(lat=latitude, lon=longitude))
    return fig

#==============================================================================

# AIS data matching
@app.callback(
    Output('sar-ais-match-table', 'data'),
    Input('base-map', 'clickData'),
    [State('data-table', 'data'),
    State('frame-date', 'children')]
)
def display_closest_match_table(clickData, data, frame_date):
    if clickData is None:
        return [{}]
    else:
        point_data = clickData['points'][0]
        latitude = point_data.get('customdata', [None])[0]
        longitude = point_data.get('customdata', [None])[1]
        name = point_data.get('customdata', [None])[3]
        timestamp = point_data.get('customdata', [None])[5]
        prediction = point_data.get('customdata', [None])[7]
        click_data = [{
            "latitude": latitude,
            "longitude": longitude,
            "name": name,
            "timestamps": timestamp,
            "prediciton": prediction},
        ]
        df_click = pd.DataFrame(click_data)
        df_results = pd.DataFrame(data)
        mask_date = df_results['date'] == frame_date
        mask_ais = df_results['prediction'] == 'Not available (AIS)'
        df_ais = df_results[mask_date & mask_ais].copy()
        if prediction == 'STS':
            lat, lon = float(df_click.iloc[0]['latitude']), float(df_click.iloc[0]['longitude'])
            df_ais['distance'] = df_ais.apply(
                lambda row: distance.distance(
                    (lat, lon), (row['latitude'], row['longitude'])
                    ).km, axis=1
                )
            df_ais['distance'] = df_ais['distance'].round(3)
            df_ais = df_ais.sort_values(by='distance')
            df_ais = df_ais.head(2).reset_index(drop=True)
            df_ais = df_ais[['distance', 'mmsi', 'name', 'country', 'timedelta']]
            df_ais.columns = df_ais.columns.str.capitalize()
            df_ais = df_ais.rename(
                columns={'Distance': 'Distance (km)', 'Mmsi': 'MMSI', 'Name': 'Ship Name'}
                )
            return df_ais.to_dict('records')
        elif prediction == 'Ship':
            lat, lon = float(df_click.iloc[0]['latitude']), float(df_click.iloc[0]['longitude'])
            df_ais['distance'] = df_ais.apply(
                lambda row: distance.distance(
                    (lat, lon), (row['latitude'], row['longitude'])
                    ).km, axis=1
                )
            df_ais['distance'] = df_ais['distance'].round(3)
            df_ais = df_ais.sort_values(by='distance')
            df_ais = df_ais.head(1).reset_index(drop=True)
            df_ais = df_ais[['distance', 'mmsi', 'name', 'country', 'timedelta']]
            # df_ais.columns = df_ais.columns.str.capitalize()
            df_ais = df_ais.rename(
                columns={
                    'distance': 'Distance (km)',
                    'mmsi': 'MMSI',
                    'name': 'Ship Name',
                    'country': 'Country',
                    'timedelta': 'Time Delta'
                    }
                )
            return df_ais.to_dict('records')
        
    # return click_data


#==============================================================================

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
        lat = point_data.get('customdata', [None])[0]
        lat_formatted = decimal_to_dms_latitude(lat)
        lon = point_data.get('customdata', [None])[1]
        lon_formatted = decimal_to_dms_longitude(lon)
        mmsi = point_data.get('customdata', [None])[2]
        name = point_data.get('customdata', [None])[3]
        country = point_data.get('customdata', [None])[4]
        timestamp = point_data.get('customdata', [None])[5]
        timedelta = point_data.get('customdata', [None])[6]
        prediction = point_data.get('customdata', [None])[7]
        # image = point_data.get('customdata', [None])[8]
        data = [
            {"Feature": "Timestamp", "Value": timestamp},
            {"Feature": "Prediction", "Value": prediction},
            {"Feature": "Latitude", "Value": lat_formatted},
            {"Feature": "Longitude", "Value": lon_formatted},
            {"Feature": "MMSI", "Value": mmsi},
            {"Feature": "Ship Name", "Value": name},
            {"Feature": "Country", "Value": country},
            {"Feature": "Time Delta", "Value": timedelta},
            # {"Feature": "Image", "Value": image}
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
            image = point_data.get('customdata', [None])[8]
            return image
        except IndexError:
            return ''

#==============================================================================



if __name__ == '__main__':
    app.run_server(debug=True)
