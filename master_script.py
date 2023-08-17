"""

Utility script to make our experiments easier 


GROUND TRUTH: 

python /home/prateiksinha/ClimaX/src/climax/global_forecast/test.py \
    --config /home/prateiksinha/ClimaX/configs/global_forecast_climax.yaml \ <-- EXPERIMENT W/WOUT THIS
    --trainer.strategy=ddp \
    --trainer.devices=1 \
    --trainer.max_epochs=50 \
    --data.root_dir=/home/prateiksinha/test_new2 \
    --data.predict_range= 2\
    --data.out_variables=['2m_temperature'] \
    --data.batch_size=1 \
    --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' \
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
        self.data = "/home/prateiksinha/new_data/processed/mpi" if (user == "P") else "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg_npz/test_new"
        self.hours = 24
        self.out_vars = ['2m_temperature']
        # self.out_vars = ['temperature']
        self.batch_size = 1 


def one_run(params): 

    cmd = f"""python /home/{params.user}/ClimaX/src/climax/global_forecast/test.py \
        --config /home/{params.user}/ClimaX/configs/global_forecast_climax.yaml \
        --trainer.strategy=ddp \
        --trainer.devices=1 \
        --trainer.max_epochs=50 \
        --data.root_dir={params.data}\
        --data.predict_range={str(params.hours)}\
        --data.out_variables={str(params.out_vars)} \
        --data.batch_size={str(params.batch_size)} \
        --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt' \
        --model.lr=5e-7 \
        --model.beta_1="0.9" \
        --model.beta_2="0.99" \
        --model.weight_decay=1e-5"""

    # Run the command 

    #args = 

    #res = subprocess.call(cmd, args)
    res = os.system(cmd)

    # Parse the output to extract what's relevant 

    # Store + Return relevant results

    return res 


def main(): 
    params = Params("P") # Advit's params
    res = one_run(params)

    print(res)


if __name__=='__main__': 
    main() 