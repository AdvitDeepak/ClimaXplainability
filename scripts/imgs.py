import os 
import json 
import numpy as np 
import re 

import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


DIR = '/home/advit/aug30_exps/all_vars'
OUT_PRED = '/home/advit/aug30_exps/all_vars/pred_kelvin'
OUT_TRUTH = '/home/advit/aug30_exps/all_vars/truth'

#VARIABLES_TO_TRY = ['temp_50', 'temp_250', 'temp_500', 'temp_600', 'temp_700', 'temp_850', 'temp_925']
#VARIABLES_TO_TRY = ['2m_temp', '10m_u__wind', '10m_v_wind', 'geo_50']


def plot(data, title, d_type, scale):
        temperature = data
        fig = plt.figure(figsize=(20,9))
        map = Basemap(projection='cyl', resolution = 'i', area_thresh = 0.3, llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90.0, anchor = 'SW')
        map.drawcoastlines()
        map.drawcountries()
        map.drawparallels(np.arange(-90,90,15.0),labels=[1,1,0,1])
        map.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1])

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
        plt.title(title, 
            {
                'fontsize' : 30
            }
        )
        #fig.show()
        if d_type == "pred": 
            plt.savefig(os.path.join(OUT_PRED, title))
        elif d_type == "truth":
            plt.savefig(os.path.join(OUT_TRUTH, title))
        
        plt.close() 
 
        

# Get a list of JSON files in alphabetical order
json_files = sorted([f for f in os.listdir(DIR) if f.endswith('.json')])
pattern = r'(\d+)\.json'

START_AT = -1

# Loop through the JSON files
for json_file in json_files:
    json_path = os.path.join(DIR, json_file)
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
        print("File Name:", json_file)

        # Define regular expressions to extract numbers
        invars_number = re.search(r'invars_(\d+)', json_file).group(1)
        hrs_number = re.search(r'hrs_(\d+)', json_file).group(1)

        # Convert extracted strings to integers
        invars_number = int(invars_number)
        hrs_number = hrs_number
            
        data = data[0]
        output = np.array(data['output']).squeeze()
        prediction = np.array(data['prediction']).squeeze()

        fig_title = f"pred_at_{hrs_number}_hrs"
        min_range = min([np.amin(prediction), np.amin(output)]) 
        max_range = max([np.amax(prediction), np.amax(output)]) 
        scale = (min_range, max_range)
        plot(prediction, fig_title, "pred", scale)
        #plot(output, f"truth_{numerical_part}_hrs", "truth") 

       