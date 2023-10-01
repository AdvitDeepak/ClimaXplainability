
import os
import glob 
import xarray as xr
import numpy as np
import re 
import netCDF4 as nc
import pprint 

from ClimaXplainability.scripts.npz_parsing.regrid import regrid_a_file 




class DataParser(): 

    def __init__(self, file_paths, climate_models, year_range, variables, degree): 
        self.file_paths = file_paths
        self.climate_models = climate_models 
        year_start, year_end = year_range 
        self.year_range = range(year_start, year_end)
        self.variables = variables 
        self.degree = degree 

        self.LAT_FOR_DGR = 180 / degree 

        #self.pp = pprint.PrettyPrinter(indent=4, width=60)
    
    def execute(self, verbose=True): 
      
        """
        1) Split nc files, if needed
        2) Regrid nc files, if needed 
        3) Run nc_to_np w/ train split
        4) Run nc_to_np w/ test split 
        5) Remove everything in train folder 
        """

        print("\n=== Entering NC collection ===\n")
        # 1) Get all .nc files 
        self.all_nc_files = {} 

        for model in self.climate_models:
            self.all_nc_files[model] = {} 

            for variable in self.variables:
                self.all_nc_files[model][variable] = {} 
                model_variable_path = os.path.join(self.file_paths['download_raw'], model, variable)
                
                if os.path.exists(model_variable_path):
                    print(f"Files for {model} - {variable}:")
                    nc_files = [filename for filename in os.listdir(model_variable_path) if filename.endswith('.nc')]
                    
                    for nc_file in nc_files:
                        print(f" - {nc_file}")
                        # Use regular expression to find the year
                        match = re.search(r'\d{4}', nc_file)

                        if match:
                            year = int(match.group())
                            print("   - Year:", year)
                        else:
                            print("   - Year not found in the filename.")
                            year = int(input("   - Please enter the 4 digit year, and press enter:"))

                        self.all_nc_files[model][variable][year] = {}
                        self.all_nc_files[model][variable][year][nc_file] = {}  
                else:
                    print(f"Directory not found for {model} - {variable}")
        

        print("\n=== Entering Splitting Step ===\n")
        # 2) Ensure all .nc files are split 
        pass 

        print("\n=== Entering Regridding Step ===\n")
        # 3) Regrid nc files, if needed 
        for model in self.all_nc_files: 
            for variable in self.all_nc_files[model]: 
                for year in self.all_nc_files[model][variable]: 
                    for file_name in self.all_nc_files[model][variable][year]: 
                        file_path = os.path.join(self.file_paths['download_raw'], model, variable, file_name)

                        print(f" - Checking .nc file -- {file_name}")
                        dims_and_vars = self.get_nc_dims(file_path)
                        self.all_nc_files[model][variable][year][file_name] = dims_and_vars
                        
                        lat = dims_and_vars['lat'][0]
                        if lat != self.LAT_FOR_DGR: 
                            print(f"   - Need to regrid! Current lat is {lat}, need {self.LAT_FOR_DGR}")
                            
                            save_path = os.path.join(self.file_paths['download_raw'], model, variable, f"regridded_{file_name}")
                            regrid_a_file(file_path, save_path, self.degree)
                            print(f"   - Succesfully regridded file.")
                            dims_and_vars = self.get_nc_dims(file_path)
                            self.all_nc_files[model][variable][year][file_name] = dims_and_vars
                            print(f"   - Deleting original file")
                            try:
                                os.remove(file_path)
                                print(f"     - File deleted successfully.")
                            except OSError as e:
                                print(f"     - Error deleting the file", e)



        print("\n=== Entering Conversion Step ===\n")
        # 4) Converting to NPZ file 

        for model in self.climate_models: 
            root_dir = f"{self.file_paths['download_raw']}/{model}"
            save_dir = f"{self.file_paths['processed_new']}/{model}"

            partition = "train"
            self.nc2np(root_dir, self.variables, self.year_range, save_dir, partition)
            
            partition = "test"
            self.nc2np(root_dir, self.variables, self.year_range, save_dir, partition)

            self.get_lat_long(root_dir, self.variables, self.year_range, save_dir)


        print("\n=== Finished Execution. ===\n")
        #self.pp.pprint(self.all_nc_files)
        #print(self.all_nc_files)
    


    def get_nc_dims(self, path): 
        nc_file = nc.Dataset(path, 'r')
        dims_and_vars = {} 

        for dim_name in nc_file.dimensions.keys():
            dim_size = len(nc_file.dimensions[dim_name])
            #print(f"{dim_name}: {dim_size}")
            dims_and_vars[dim_name] = dim_size

        for var_name in nc_file.variables.keys():
            var_shape = nc_file.variables[var_name].shape
            #print(f"{var_name}: {var_shape}")
            dims_and_vars[var_name] = var_shape 

        nc_file.close()

        return dims_and_vars




    def get_lat_long(self, path, variables, years, save_dir): 
        ps = glob.glob(os.path.join(path, variables[0], f"*{years[0]}*.nc"))
        x = xr.open_mfdataset(ps[0], parallel=True)
        lat = np.array(x["lat"])
        lon = np.array(x["lon"])
        np.save(os.path.join(save_dir, "lat.npy"), lat)
        np.save(os.path.join(save_dir, "lon.npy"), lon)



    
    def nc2np(self, path, variables, years, save_dir, partition, num_shards_per_year=1):
        os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

        if partition == "train":
            normalize_mean = {}
            normalize_std = {}
        climatology = {}

        constants_path = os.path.join(path, "constants.nc")
        constants_are_downloaded = os.path.isfile(constants_path)

        if constants_are_downloaded:
            print('FOUND CONSTANTS')
            constants = xr.open_mfdataset(
                constants_path, combine="by_coords", parallel=True
            )
            constant_fields = [Constants.VAR_TO_NAME[v] for v in Constants.CONSTANTS if v in Constants.VAR_TO_NAME.keys()]
            constant_values = {}
            for f in constant_fields:
                constant_values[f] = np.expand_dims(
                    constants[Constants.NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)
                ).repeat(Constants.HOURS_PER_YEAR, axis=0)
                if partition == "train":
                    normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
                    normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

        for year in years:
            np_vars = {}

            # constant variables
            if constants_are_downloaded:
                for f in constant_fields:
                    np_vars[f] = constant_values[f]

            # non-constant fields
            for var in variables:
                print(os.path.join(path, var, f"*{year}*.nc"))
                ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
                ds = xr.open_mfdataset(
                    ps, combine="by_coords", parallel=True
                )  # dataset for a single variable
                code = Constants.NAME_TO_VAR[var]

                if len(ds[code].shape) == 3:  # surface level variables
                    ds[code] = ds[code].expand_dims("val", axis=1)
                    # remove the last 24 hours if this year has 366 days
                    if code == "tp":  # accumulate 6 hours and log transform
                        tp = ds[code].to_numpy()
                        tp_cum_6hrs = np.cumsum(tp, axis=0)
                        tp_cum_6hrs[6:] = tp_cum_6hrs[6:] - tp_cum_6hrs[:-6]
                        eps = 0.001
                        tp_cum_6hrs = np.log(eps + tp_cum_6hrs) - np.log(eps)
                        np_vars[var] = tp_cum_6hrs[-Constants.HOURS_PER_YEAR:]
                    else:
                        np_vars[var] = ds[code].to_numpy()[-Constants.HOURS_PER_YEAR:]

                    if partition == "train":
                        # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                        if var not in normalize_mean:
                            normalize_mean[var] = [var_mean_yearly]
                            normalize_std[var] = [var_std_yearly]
                        else:
                            normalize_mean[var].append(var_mean_yearly)
                            normalize_std[var].append(var_std_yearly)

                    clim_yearly = np_vars[var].mean(axis=0)
                    if var not in climatology:
                        climatology[var] = [clim_yearly]
                    else:
                        climatology[var].append(clim_yearly)

                else:  # pressure-level variables
                    assert len(ds[code].shape) == 4
                    all_levels = ds["level"][:].to_numpy()
                    all_levels = np.intersect1d(all_levels, Constants.DEFAULT_PRESSURE_LEVELS)
                    for level in all_levels:
                        ds_level = ds.sel(level=[level])
                        level = int(level)
                        # remove the last 24 hours if this year has 366 days
                        np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[
                            -Constants.HOURS_PER_YEAR:
                        ]

                        if partition == "train":
                            # compute mean and std of each var in each year
                            var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                            var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                            if f"{var}_{level}" not in normalize_mean:
                                normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                                normalize_std[f"{var}_{level}"] = [var_std_yearly]
                            else:
                                normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                                normalize_std[f"{var}_{level}"].append(var_std_yearly)

                        clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                        if f"{var}_{level}" not in climatology:
                            climatology[f"{var}_{level}"] = [clim_yearly]
                        else:
                            climatology[f"{var}_{level}"].append(clim_yearly)

            assert Constants.HOURS_PER_YEAR % num_shards_per_year == 0
            num_hrs_per_shard = Constants.HOURS_PER_YEAR // num_shards_per_year
            for shard_id in range(num_shards_per_year):
                start_id = shard_id * num_hrs_per_shard
                end_id = start_id + num_hrs_per_shard
                sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
                np.savez(
                    os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                    **sharded_data,
                )

        if partition == "train":
            for var in normalize_mean.keys():
                if not constants_are_downloaded or var not in constant_fields:
                    normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                    normalize_std[var] = np.stack(normalize_std[var], axis=0)

            for var in normalize_mean.keys():  # aggregate over the years
                if not constants_are_downloaded or var not in constant_fields:
                    mean, std = normalize_mean[var], normalize_std[var]
                    # var(X) = E[var(X|Y)] + var(E[X|Y])
                    variance = (
                        (std**2).mean(axis=0)
                        + (mean**2).mean(axis=0)
                        - mean.mean(axis=0) ** 2
                    )
                    std = np.sqrt(variance)
                    # E[X] = E[E[X|Y]]
                    mean = mean.mean(axis=0)
                    normalize_mean[var] = mean
                    if var == "total_precipitation":
                        normalize_mean[var] = np.zeros_like(normalize_mean[var])
                    normalize_std[var] = std

            np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
            np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

        for var in climatology.keys():
            climatology[var] = np.stack(climatology[var], axis=0)
        climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
        np.savez(
            os.path.join(save_dir, partition, "climatology.npz"),
            **climatology,
        )



"""

Helper "Class" (Struct) to Store Constants 

"""

class Constants: 
    HOURS_PER_YEAR = 8736  # 8760 --> 8736 which is dividable by 16
    NAME_TO_VAR = {
        "2m_temperature": "tas",
        # "2m_temperature": "t2m", <-- Original! 
        "10m_u_component_of_wind": "uas",
        #"10m_u_component_of_wind": "u10", <-- Original! 
        "10m_v_component_of_wind": "v10",
        "mean_sea_level_pressure": "msl",
        "surface_pressure": "sp",
        "toa_incident_solar_radiation": "tisr",
        "total_precipitation": "tp",
        "land_sea_mask": "lsm",
        "orography": "orography",
        "lattitude": "lat2d",
        "geopotential": "z",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "temperature": "t",
        "relative_humidity": "r",
        "specific_humidity": "q",
        "vorticity": "vo",
        "potential_vorticity": "pv",
        "total_cloud_cover": "tcc",
    }
    VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}



"""

Sample run, if script directly invoked 

"""

if __name__=='__main__': 

    FILE_PATHS = {
        'download_raw' : '/home/advit/ClimateData/download_raw', 
        'processed_new' : '/home/advit/ClimateData/processed_new',
    }

    CLIMATE_MODELS = ['AWI']
    YEAR_RANGE = (1990, 1994)
    VARIABLES = ['2m_temperature']
    DEGREE = 5.625

    d = DataParser(FILE_PATHS, CLIMATE_MODELS, YEAR_RANGE, VARIABLES, DEGREE)
    d.execute() # Can toggle verbosity on/off (def: True)