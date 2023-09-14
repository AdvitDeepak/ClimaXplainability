INITIAL_STATE = {
    "selected_mode" : False,
    "selected_npz" : [], 
    "selected_inp_var" : [], 
    "selected_out_var" : [], 
    "selected_start" : [], 
    "selected_leads" : [], 
    "run_json_dir" : [], 
    "selected_run" : [],  

    "run_df" : None, 
    "metrics_df" : None, 
    "plotly_truth" : None, 
    "plotly_pred" : None, 


}

SIDEBAR = {
    'caption' : """Welcome to ClimaXplainability! If you already have a run you wish to visualize, 
                          please turn off the 'Run + Visualize' toggle. Otherwise, leave the option on. 
                          Happy exploring! 🥳""", 

    'defaults' : {
        'npz_path' : 'localhome/data/datasets/climate/era5/1.40625_npz/', 
        'inp_vars' : None, 
        'out_vars' : None,
        "run_dir_path" :  '/localhome/advit/sep7_exps/all_vars',
    }
}

# Assuming 1.4 (NOT supporting 5.6 res for now)

CONFIG = {
    "all_vars" :  [
          "land_sea_mask",
          "orography",
          "lattitude",
          "2m_temperature",
          "10m_u_component_of_wind",
          "10m_v_component_of_wind",
          "geopotential_50",
          "geopotential_250",
          "geopotential_500",
          "geopotential_600",
          "geopotential_700",
          "geopotential_850",
          "geopotential_925",
          "u_component_of_wind_50",
          "u_component_of_wind_250",
          "u_component_of_wind_500",
          "u_component_of_wind_600",
          "u_component_of_wind_700",
          "u_component_of_wind_850",
          "u_component_of_wind_925",
          "v_component_of_wind_50",
          "v_component_of_wind_250",
          "v_component_of_wind_500",
          "v_component_of_wind_600",
          "v_component_of_wind_700",
          "v_component_of_wind_850",
          "v_component_of_wind_925",
          "temperature_50",
          "temperature_250",
          "temperature_500",
          "temperature_600",
          "temperature_700",
          "temperature_850",
          "temperature_925",
          "relative_humidity_50",
          "relative_humidity_250",
          "relative_humidity_500",
          "relative_humidity_600",
          "relative_humidity_700",
          "relative_humidity_850",
          "relative_humidity_925",
          "specific_humidity_50",
          "specific_humidity_250",
          "specific_humidity_500",
          "specific_humidity_600",
          "specific_humidity_700",
          "specific_humidity_850",
          "specific_humidity_925"
        ]
}