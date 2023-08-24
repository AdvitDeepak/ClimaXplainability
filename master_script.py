"""

Utility script to make our experiments easier 


GROUND TRUTH: 

python /home/prateiksinha/ClimaX/src/climax/global_forecast/test.py \
    --config /home/prateiksinha/ClimaX/configs/global_forecast_climax`.yaml \ <-- EXPERIMENT W/WOUT THIS
    --trainer.strategy=ddp \
    --trainer.devices=1 \
    --trainer.max_epochs=50 \
    --data.root_dir=/home/prateiksinha/test_new2 \
    --data.predict_range= 2\
    --data.out_variables=['2m_temperature'] \
    --data.batch_size=1 \
    --model.pretrained_path='https://huggingface.co/tungnd/ClimaX/resolve/main/5.625deg.ckpt' \
    --model.lr=5e-7 \
    --model.beta_1="0.9" \
    --model.beta_2="0.99" \
    --model.weight_decay=1e-5


"""

import os 
import subprocess 


class Params(): 
    def __init__(self, user):
        self.user = "prateiksinha" if (user == "P") else "advit"
        print(f"User is {self.user}")
        if user == "P": 
            self.data = "/home/prateiksinha/new_data/processed/mpi" 
        else: 
            self.data = "/home/advit/ClimateData/processed_new/AWI"

        self.hours = 24
        self.out_vars = ['2m_temperature']
        self.batch_size = 1 

def one_run(params): 
    cmd = f"""python /home/{params.user}/ClimaX/src/climax/global_forecast/test.py \
        --config '/home/{params.user}/ClimaX/configs/global_forecast_climax.yaml' \
        --trainer.strategy=ddp \
        --trainer.devices=1 \
        --trainer.max_epochs=50 \
        --trainer.default_root_dir=/home/{params.user}/ClimaX/exps/global_forecast_climax \
        --data.root_dir={params.data}\
        --data.predict_range={str(params.hours)}\
        --data.out_variables={str(params.out_vars)} \
        --data.batch_size={str(params.batch_size)} \
        --model.pretrained_path='https://huggingface.co/tungnd/ClimaX/resolve/main/5.625deg.ckpt' \
        --model.lr=5e-7 \
        --model.beta_1="0.9" \
        --model.beta_2="0.99" \
        --model.weight_decay=1e-5"""

    res = os.system(cmd)
    return res 

def one_run_json(json, params):
    cmd = f"""python /home/{params.user}/ClimaX/src/climax/global_forecast/test.py \
        --config /home/{params.user}/ClimaX/configs/global_forecast_climax.yaml \
        --trainer.strategy=ddp \
        --trainer.devices=1 \
        --trainer.max_epochs=50 \
        --trainer.default_root_dir=/home/{params.user}/ClimaX/exps/global_forecast_climax` \
        --data.root_dir={json.climate_model_init}\
        --data.predict_range={str(json.lead_time_in_hrs)}\
        --data.out_variables={str(json.out_vars)} \
        --data.batch_size={str(params.batch_size)} \
        --model.pretrained_path='https://huggingface.co/tungnd/ClimaX/resolve/main/5.625deg.ckpt' \
        --model.lr=5e-7 \
        --model.beta_1="0.9" \
        --model.beta_2="0.99" \
        --model.weight_decay=1e-5"""

    res = os.system(cmd)
    return res 


def run_json(json):
    for i in json:
        one_run_json(i)


def main(): 
    params = Params("P") # Advit's params
    res = one_run(params)

    print("About to print results!")
    print(res)


if __name__=='__main__': 
    main() 