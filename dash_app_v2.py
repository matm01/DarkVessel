from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import os
import pandas as pd
import plotly.express as px


app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
load_figure_template('SPACELAB')


# Timestamps for existing SAR images
timestamp_file = 'data/timestamps_sar_images.csv'
df_timestamp = pd.read_csv(timestamp_file)
df_timestamp = df_timestamp.set_index('TILE_ID')
unique_dates = df_timestamp['DATE'].unique()
list_of_unique_dates = unique_dates.tolist()

# Results from the predictions
results_file = 'results/sample_results.csv'
df_results = pd.read_csv(results_file)

# Map
token = os.environ.get('MAPBOX_TOKEN')
latitude = 36.53353  # 36.34289 
longitude = 22.721728  # 22.43289


#==============================================================================
# Define the content of the app
#==============================================================================

header = dbc.Row([
    dbc.Col([
        html.H1("Ship-to-Ship (STS) Transfer Detection in SAR images", id='header')
    ])
])

sidebar = dbc.Row([
    dbc.Col([
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with navigation links", className="lead"
        ),
        dcc.Dropdown(
            id='start-date',
            options=[{'label': date, 'value': date} for date in list_of_unique_dates],
            value=list_of_unique_dates[0] if list_of_unique_dates else None, # Sets the default value to the first date
        ),
        dcc.Dropdown(
            id='end-date',
            options=[{'label': date, 'value': date} for date in list_of_unique_dates],
            value=list_of_unique_dates[1] if list_of_unique_dates else None, # Sets the default value to the first date
        ),
        dbc.Button("Filter", id='filter-button', n_clicks=0),
        dcc.Store(id='data-table')
    ])
])

control = dbc.Row([
    dbc.Col(dbc.ButtonGroup([
        dbc.Button('Previous', id='prev-btn', n_clicks=0),
        dbc.Button('Next', id='next-btn', n_clicks=0),
    ]), width=2),
    dbc.Col(html.Div(id='frame-date'), width=2),
])


interactive_map = dbc.Row([
    dbc.Col([
        dcc.Graph(
            id='base-map',
            config={'displayModeBar': False},
            # clickData=None
        )
    ])
])



#==============================================================================
# Define the layout of the app
#==============================================================================


app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(
                header,
                width=12,
                style={'textAlign': 'left', "background-color": "#f8f9fa", "padding": "1rem 1rem"}
            ),
        ]),
        # Sidebar
        dbc.Row([
            dbc.Col(
                sidebar,
                width=2
            ),
            dbc.Col([
                control,
                interactive_map,
                ],width=8
            ),
            # dbc.Col(
            #     dbc.Row([
            #         dbc.Col(dbc.Button('Previous', id='prev-btn', n_clicks=0), width=1),
            #         dbc.Col(dbc.Button('Next', id='next-btn', n_clicks=0), width=1),
            #         dbc.Col(html.Div(id='frame-date'), width=1),
            #     ]), width=2

            # ),
        ]),
    ], fluid=True
)

#==============================================================================

# Update end date options based on start date
@app.callback(
    Output('end-date', 'options'),
    Input('start-date', 'value')
)
def update_end_date_options(start_date):
    if start_date:
        # Ensure the end date is not before the start date
        return [{'label': date, 'value': date} for date in list_of_unique_dates if date > start_date]
    else:
        # If no start date is selected, allow any date
        return [{'label': date, 'value': date} for date in list_of_unique_dates]

#==============================================================================

# Filter results based on date range
@app.callback(
    Output('data-table', 'data'),
    Input('filter-button', 'n_clicks'),
    State('start-date', 'value'),
    State('end-date', 'value')
)
def update_data_table(n_clicks, start_date, end_date):
    if start_date and end_date:
        mask_dates = (df_results['DATE'] >= start_date) & (df_results['DATE'] <= end_date)
        filtered_df = df_results[mask_dates]
        filtered_df = filtered_df.sort_values('DATE')
        data = filtered_df.to_dict('records')
        # print(data)
        return data
    else:
        data = df_results.to_dict('records')
        return None

#==============================================================================

# Update date for map based on frame selection using previous and next buttons
@app.callback(
    Output('frame-date', 'children'),
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('start-date', 'value'),
    State('end-date', 'value'),
    State('frame-date', 'children')]
)
def update_map_date(prev_clicks, next_clicks, start_date, end_date, frame_date):
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
        print(filtered_dates[new_index])
        print(filtered_dates)
        return filtered_dates[new_index]

#==============================================================================

# Update map with predictions
@app.callback(
    Output('base-map', 'figure'),
    Input('frame-date', 'children'),
    State('data-table', 'data')
)
def update_map(frame_date, data):
    df_results = pd.DataFrame(data)
    mask_date = df_results['DATE'] == frame_date
    df_results = df_results[mask_date]
    if data:
        # print(f'Updating map on clicks with {data}')
        # # Convert data to DataFrame for easier manipulation
        fig = px.scatter_mapbox(
            df_results,
            lat="latitude",
            lon="longitude",
            color="prediction",
            zoom=9,
            height=700,
            # mapbox_style="carto-positron",
            hover_data=['name', 'latitude', 'longitude', 'prediction', 'image']
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




if __name__ == '__main__':
    app.run_server(debug=True)
