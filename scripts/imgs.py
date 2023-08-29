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


DIR = 'output_jsons'
OUT_PRED = 'pred_imgs'
OUT_TRUTH = 'truth_imgs'


def plot(data, title, d_type):
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
            shading='flat'
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

START_AT = 15 

# Loop through the JSON files
for json_file in json_files:
    json_path = os.path.join(DIR, json_file)
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
        print("File Name:", json_file)
        match = re.search(pattern, json_file)
        if match:
            int_match = int(match.group(1)) 
            if int_match < START_AT: 
                 print(f" - Skipping {int_match}, already exists.")
                 continue 
            
            numerical_part = match.group(1)
            data = data[0]
            output = np.array(data['output']).squeeze()
            prediction = np.array(data['prediction']).squeeze()

            plot(prediction, f"pred_{numerical_part}_hrs", "pred")
            plot(output, f"truth_{numerical_part}_hrs", "truth") 

       