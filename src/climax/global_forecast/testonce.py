from torchvision.transforms import transforms
from typing import Any
import torch 
from torch.utils.data import DataLoader, IterableDataset

from climax.arch import ClimaX
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from climax.utils.pos_embed import interpolate_pos_embed

from climax.pretrain.datamodule import collate_fn
from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)

import numpy as np 
import os 
from pytorch_lightning import LightningModule

from datetime import datetime, timedelta

# ROOT_DIR = "/home/advit/ClimateData/processed_new/AWI"
# ROOT_DIR = "/home/prateiksinha/new_data/processed/awi"
ROOT_DIR = "/home/advit/test_new2/"

class ModifiedGlobalForecastModule(LightningModule): 

    def __init__(
        self,
        net: ClimaX,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        ##self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)


    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_test_clim(self, clim):
        self.test_clim = clim

    def test_step(self, batch: Any, batch_idx: int):
        print(f"(module.py) Entered test_step function w/ batch_idx {batch_idx}")
        x, y, lead_times, variables, out_variables = batch
        print("X", x)
        print("Y", y)
        print("LEAD", lead_times)
        print("IN VARS", variables)
        print("OUT VARS", out_variables)


        self.pred_range = 1

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        
        print(f"(module.py) About to call self.net.evaluate() function")
        all_loss_dicts, json = self.net.evaluate(
            x=x,
            y=y,
            lead_times=lead_times,
            variables=variables,
            out_variables=out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]
        return loss_dict, json
    

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_normalize(variables=['2m_temperature']):
    # print(variables)
    normalize_mean = dict(np.load(os.path.join(ROOT_DIR, "normalize_mean.npz")))
    mean = []
    for var in variables:
        if var != "total_precipitation":
            mean.append(normalize_mean[var])
        else:
            mean.append(np.array([0.0]))
    normalize_mean = np.concatenate(mean)
    normalize_std = dict(np.load(os.path.join(ROOT_DIR, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[var] for var in variables])
    return transforms.Normalize(normalize_mean, normalize_std)


def get_lat_lon():
    lat = np.load(os.path.join(ROOT_DIR, "lat.npy"))
    lon = np.load(os.path.join(ROOT_DIR, "lon.npy"))
    return lat, lon

def get_climatology(partition, variables):
    path = os.path.join(ROOT_DIR, partition, "climatology.npz")
    clim_dict = np.load(path)

    clim = np.concatenate([clim_dict[var] for var in variables])
    clim = torch.from_numpy(clim)
    return clim


def days_in_year(year=datetime.now().year):
    import calendar
    return 365 + calendar.isleap(int(year))

def year_to_days_since_1850(year = None, partition = None, partitions_per_year = 12):
    
    reference_date = datetime(1850, 1, 1, 0, 0, 0)
    new_date = datetime(int(year), 1, 1, 0, 0, 0)
    year_in_days = (new_date - reference_date).days

    if partition is None:
        return year_in_days
    else:
        return year_in_days + ((int(partition) / partitions_per_year) * days_in_year(year))

def append_to_json(j, file_name, partitions_per_year):
    j['climate_model_init'] = ROOT_DIR.split('/')[-1]
    year, partition = tuple(file_name.split('/')[-1].strip('.npz').split('_'))
    j['days_since_1850'] = year_to_days_since_1850(year, partition, partitions_per_year)
    return j


"""

The heart of our testonce.py script, the run() method. 

"""


def run(npz_path, lead_time, in_variables, out_variables, out_dir):
    pretrained_path = 'https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' 

    default_vars = [
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
        "specific_humidity_925",
        ]

    mod = ModifiedGlobalForecastModule(ClimaX(default_vars), pretrained_path) 

    our_transforms = get_normalize()
    our_output_transforms = get_normalize(out_variables)
    
    normalization = our_output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    mod.set_denormalization(mean_denorm, std_denorm)
    
    mod.set_lat_lon(*get_lat_lon())
    mod.set_pred_range(1) #TODO - What is this? Is 1 okay? 

    clim = get_climatology("test", out_variables) # TODO - is this in_variables or out_variables? 
    mod.set_test_clim(clim)

    file_list = [npz_path]
    data_test = IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=file_list,
                            start_idx=0,
                            end_idx=1,
                            variables=in_variables,
                            out_variables=out_variables,
                            shuffle=False,
                            multi_dataset_training=False,
                        ),
                        # max_predict_range=lead,
                        max_predict_range= lead_time,
                        random_lead_time=False,
                        hrs_each_step=1,
                    ),
                    transforms=our_transforms,
                    output_transforms=our_output_transforms,
                )


    X = DataLoader(
                data_test,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=1,
                pin_memory=False,
                collate_fn=collate_fn,
            )

    loss = ""
    idx = 0 
    batch = None 

    final_json = []
    # first = True
    for batch in X: 
        print(idx)
        idx += 1
        batch = batch
        loss, output_json = mod.test_step(batch, idx)
        final_json.append(append_to_json(output_json, file_list[0], 12))

        print(loss)
        break

    import json
    from time import time
    with open(f"{out_dir}/final_invars_{len(in_variables)}_hrs_{(lead_time):04}.json", "w") as outfile:
        json.dump(final_json, outfile)


"""

The main testing script. 
Currently seeing smoothness 

"""

if __name__=='__main__': 

    BASE_VARIABLES = ['2m_temperature']
    #VARIABLES_TO_TRY = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential_50', 'temperature_50', 'temperature_250']
    VARIABLES_TO_TRY = ['temperature_50', 'temperature_250', 'temperature_500', 'temperature_600']

    DUMP_DIRECTORY = r"/home/prateiksinha/ClimaX/aug_31_jsons"
    NPZ_PATH = r"/home/prateiksinha/test_new2/test/2017_0.npz"


    # Run ClimaX with each subset of variables

    for i in range(len(VARIABLES_TO_TRY)):
        in_variables = BASE_VARIABLES + VARIABLES_TO_TRY[0:i]
        out_variables = BASE_VARIABLES

        #LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
        LEAD_TIMES = [6] 

        print(f"Running experiment w/ in_vars: {in_variables}")

        for time in LEAD_TIMES: 
            print(f" - Lead Time: {time}")
            run(npz_path=NPZ_PATH, lead_time=time, in_variables=in_variables, out_variables=out_variables, out_dir=DUMP_DIRECTORY)

        print(f"")
    
    
