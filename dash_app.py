from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from pathlib import Path
import numpy as np
import os
import src.predictions_with_land_mask as preds

ais_file = 'data/ais_datalastic_filtered.csv'
timestamp_file = 'data/timestamps_sar_images.csv'

bucket_id = os.environ.get('SAR_BUCKET')
local_path = f'data/{bucket_id}/VH'
# local_path = 'data/bucket/VH'

# Get list of SAR images from local path
list_of_imgs = os.listdir(local_path)


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(children='DarkVessel', style={'textAlign': 'center'}),
        dcc.Dropdown(list_of_imgs, list_of_imgs[0], id='dropdown-selection'),
        dcc.Graph(id='graph-content'),
    ]
)

@callback(Output('graph-content', 'figure'), Input('dropdown-selection', 'value'))
def update_graph(value):
    # Reading AIS data from datalastic dataset
    df_ais = pd.read_csv(ais_file)
    
    # Reading timestamp data from SAR images
    df_timestamp = pd.read_csv(timestamp_file)
    df_timestamp = df_timestamp.set_index('TILE_ID')
    
    # Run prediction on selected image
    df_preds = preds.predict(value, plot=False)

    # Get timestamp for selected image
    image_id = value.split('.')[0]
    image_timestamp = df_timestamp.loc[image_id, 'TIMESTAMP']
    
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
        zoom=8,
        height=800,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


if __name__ == '__main__':
    app.run(debug=True)
