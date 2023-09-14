import os 
import json 

import pandas as pd 
import numpy as np 
from tabulate import tabulate
import plotly.graph_objects as go 

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import plotly.express as px 

def find_json_files(directory):
    json_files = []
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                #json_files.append(os.path.join(root, file))
                json_files.append(file)
    return json_files


def report_new(data):
    # By default, are choosing the 0th index of the prediction (TODO: is this still needed?) 
    data = data[0]
    
    if not data['climate_model_init']: 
        data['climate_model_init'] = "Era5"

    main_data = {
        'Category': ['input variables', 'output variables', 'lead time', 'climate model', 'days since 1850'],
        'Value': [', '.join(data['variables']), ', '.join(data['out_variables']), f"{round(data['lead_times']*100, 2)} hrs", data['climate_model_init'], data['days_since_1850']]
    }

    main_df = pd.DataFrame(main_data)

    metric_dfs = []
    for i in data['metrics']:
        metric_data = [[j, data['metrics'][i][j]] for j in data['metrics'][i]]
        metric_df = pd.DataFrame(metric_data, columns=['Metric', 'Value'])
        metric_dfs.append(metric_df)

    return main_df, metric_df


def one_map(data, scale): 
    temperature = np.roll(data, data.shape[1] // 2, axis=1)

    map = Basemap(projection='cyl', resolution = 'i', area_thresh = 0.3, llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90.0, anchor = 'SW')
    map.drawcoastlines()
    map.drawcountries()
    map.drawparallels(np.arange(-90,90,15.0),labels=[1,1,0,1])
    map.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1])

    # temperature2 = upscale(temperature, fig)
    heatmap = plt.pcolormesh(
        np.linspace(-180, 180, num = temperature.shape[1]+1),
        np.linspace(-90, 90, num = temperature.shape[0]+1),
        temperature, 
        cmap='bwr',
        shading='flat',
        vmin = scale[0],
        vmax = scale[1]
    )

    map.colorbar(heatmap, pad=1)

    return heatmap 


def round_nested_integers(arr):
    if isinstance(arr, list):
        return [round_nested_integers(item) for item in arr]
    elif isinstance(arr, int):
        return round(arr, 3)
    else:
        return arr


def get_choro_fig(arr, arr2, min_range, max_range): 
    # Create a grid of latitude and longitude coordinates
    latitudes = np.linspace(90, -90, 129)
    longitudes = np.linspace(0, 360, 257)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

    rectangles_arr = []
    temperature_arr = []

    geo_path = 'rectangles_smaller.json'

    # Check if the file specified in geo_path exists
    if os.path.exists(geo_path):
        # File exists, load it as JSON
        print(f"rectangles.json exists -- thank god!")
        with open(geo_path, 'r') as file:
            rectangles = json.load(file)
    else:
        # File does not exist, create an empty GeoJSON object
        print(f"uh oh! generating json file")
        rectangles = {
            "type": "FeatureCollection",
            "features": []
        }
        round_factor = 1
        id_counter = 0
        for i in range(128):
            for j in range(256):
                lat_center = (lat_mesh[i, j] + lat_mesh[i + 1, j] + lat_mesh[i + 1, j + 1] + lat_mesh[i, j + 1]) / 4
                lon_center = (lon_mesh[i, j] + lon_mesh[i + 1, j] + lon_mesh[i + 1, j + 1] + lon_mesh[i, j + 1]) / 4

                rectangle = {
                    'type': 'Feature',
                    'properties': {
                        "Coords": f'({lat_center}, {lon_center})',
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [round(lon_mesh[i, j], round_factor), round(lat_mesh[i, j], round_factor)],
                            [round(lon_mesh[i + 1, j], round_factor), round(lat_mesh[i + 1, j], round_factor)],
                            [round(lon_mesh[i + 1, j + 1], round_factor), round(lat_mesh[i + 1, j + 1], round_factor)],
                            [round(lon_mesh[i, j + 1], round_factor), round(lat_mesh[i, j + 1], round_factor)],
                            [round(lon_mesh[i, j], round_factor), round(lat_mesh[i, j], round_factor)]
                        ]]
                    },
                    'id': f'({lat_center}, {lon_center})'
                }
                rectangles_arr.append(rectangle)
                id_counter += 1

        rectangles["features"] = rectangles_arr


        # Store the empty GeoJSON as a JSON file
        with open(geo_path, 'w') as file:
            json.dump(rectangles, file)

        print(f"wrote rectangles.json -- good rom here!")

    # TODO - have this stored somewhere 
    coord_center_arr = [] 

    for i in range(128):
        for j in range(256):
            lat_center = (lat_mesh[i, j] + lat_mesh[i + 1, j] + lat_mesh[i + 1, j + 1] + lat_mesh[i, j + 1]) / 4
            lon_center = (lon_mesh[i, j] + lon_mesh[i + 1, j] + lon_mesh[i + 1, j + 1] + lon_mesh[i, j + 1]) / 4

            coord_center_arr.append(f'({lat_center}, {lon_center})')


    print("Making list of dictionaries")
    counter = 0
    for i in range(128):
        for j in range(256):
            for idx, a in enumerate(arr): 
                temperature_arr.append({"Time" : idx, "Coords": coord_center_arr[counter], "Temp": a[i][j]})
            counter += 1    

    print("Made list of dictionaries...")
    df = pd.DataFrame(temperature_arr)
    print("Made dataframe from list...")

    # TODO - simplify rectangles geometry down by a good amount (like 1000m or smth?)

    # Plot the rectangles on a map using px.choropleth_mapbox
    fig = px.choropleth_mapbox(
        df,
        geojson=rectangles,
        locations=df.Coords,
        color="Temp",  # Use temperature data for coloring
        color_continuous_scale="RdBu_r",  # Adjust the color scale as needed
        range_color=(min_range, max_range),  # Adjust the temperature range as needed
        opacity=0.5,
        mapbox_style="carto-positron",  # Use a map style of your choice
        center={"lat": 0, "lon": 180},  # Center the map
        zoom=0,  # Adjust the initial zoom level
        labels={'Temp' : 'Temp (K)'}, 
        animation_frame='Time',
    )

    fig.update_traces(marker_line_width=0)  
    fig.update_layout(transition = {'duration': 2000})
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        height=300
    )

    return fig 



def generate_map(dir, jsons): 

    truth_arr = []
    preds_arr = [] 

    for j in jsons: 
        with open(os.path.join(dir, j)) as f: 
            data = json.load(f)
        
        # TODO: taking 0th index -- IS THIS STILL NEEDED? 
        data = data[0]

        truth_arr.append(np.array(data['output']).squeeze())
        preds_arr.append(np.array(data['prediction']).squeeze())


    # First, we determine the scale for the WHOLE plotly interactive figure (-80C to 60C)

    min_range = min([np.amin(truth_arr)]) 
    max_range = max([np.amax(truth_arr)]) 
    scale = (min_range, max_range)

    print(f"(generate_map) Scale: {scale}")


    """ FRAME CODE """

    frames = [] 
    firstFrame = None 

    # Roll all of the temperatures 

    for i in range(len(truth_arr)): 
        temperature = np.flipud(truth_arr[i])
        truth_arr[i] = np.roll(temperature, temperature.shape[1] // 2, axis=0)

        preds = np.flipud(preds_arr[i])
        preds_arr[i] = np.roll(preds, preds.shape[1] // 2, axis=0)

    print(f"(generate_map) Calling fig gen func...")
    fig1 = get_choro_fig(truth_arr, preds_arr, min_range, max_range)
    fig2 = get_choro_fig(preds_arr, truth_arr, min_range, max_range)

    return fig1, fig2