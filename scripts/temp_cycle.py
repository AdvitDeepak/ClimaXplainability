import os
import re 
import json 
import numpy as np 
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline


DIR = 'output_jsons'

# Get a list of JSON files in alphabetical order
json_files = sorted([f for f in os.listdir(DIR) if f.endswith('.json')])
pattern = r'(\d+)\.json'

STOP_AT = 94

LA_LAT = 21
LA_LONG = 42


# LA_LAT = 19
# LA_LONG = 4

LA_LAT = 22
LA_LONG = 18

LA_LAT = 16
LA_LONG = 29

input_la = [] 
output_la = [] 
pred_la = [] 

# Loop through the JSON files
for json_file in json_files:
    json_path = os.path.join(DIR, json_file)
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
        #print("File Name:", json_file)
        match = re.search(pattern, json_file)
        if match:
            int_match = int(match.group(1)) 
            if int_match > STOP_AT: 
                 print(f"Breaking")
                 break 
            
            numerical_part = match.group(1)
            data = data[0]

            output = np.array(data['output']).squeeze()
            input = np.array(data['input']).squeeze()
            prediction = np.array(data['prediction']).squeeze()
            print(input.shape)
            #print(input)
            #print(input[LA_LONG, LA_LAT])

            input_la.append(input[LA_LAT, LA_LONG])
            output_la.append(output[LA_LAT, LA_LONG])
            pred_la.append(prediction[LA_LAT, LA_LONG])

            #print(output_la)
            #print(pred_la)
            #print(input_la)
            #print() 



# Create line plot of temperature data

hours = np.arange(1, len(output_la) + 1)

plt.figure(figsize=(14, 6))
plt.plot(hours, np.array(pred_la), marker='o', linestyle='-', color='b', label='Temperature ClimaX Prediction')
plt.plot(hours, np.array(output_la), marker='s', linestyle='-', color='r', label='Temperature AWI Truth')
#plt.plot(hours, np.array(input_la), marker='^', linestyle='-', color='g', label='Temperature AWI Base')

# climax_interp = make_interp_spline(hours, np.array(pred_la), k=3)  # Cubic spline interpolation
# awi_interp = make_interp_spline(hours, np.array(output_la), k=3)  # Cubic spline interpolation

# # Create smoothed lines
# hours_smooth = np.linspace(hours.min(), hours.max(), 300)  # Generate more points for smoother curve
# climax_smooth = climax_interp(hours_smooth)
# awi_smooth = awi_interp(hours_smooth)

# plt.plot(hours_smooth, climax_smooth, linestyle='--', color='b', label='Smoothed ClimaX Prediction')
# plt.plot(hours_smooth, awi_smooth, linestyle='--', color='r', label='Smoothed AWI Truth')


plt.xlabel('Hour (Lead Time)')
plt.ylabel('Temperature (Deviation)')
plt.title('(Kwajalein, Marshall Islands) Temperature Vs. Hour - ClimaX, AWI Truth')
plt.grid(True)
plt.legend()
plt.show()
