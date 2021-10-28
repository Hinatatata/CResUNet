import os
import shutil
import pandas as pd
import torch
from config import experiment_params, data_params, model_params
from data_creator import DataCreator
from batch_generator import BatchGenerator
from experiment import predict, train
from models.baseline.convlstm import ConvLSTM
from models.baseline.u_net import UNet
from models.baseline.CRU_net_3 import CRUNet
from models.baseline.SmaAt_UNet import SmaAt_UNet

def run():

    val_ratio = experiment_params['val_ratio']  # 0.1
    test_ratio = experiment_params['test_ratio']  # 0.1
    normalize_flag = experiment_params['normalize_flag']  # True
    model_name = experiment_params['model']
    device = experiment_params['device']  # 'cuda'

    model_dispatcher = {
        'convlstm': ConvLSTM,
        'u_net': UNet,
        'cru_net' : CRUNet,
        'SmaAt_UNet':SmaAt_UNet
    }


    start_date_str = '1995-01-01'
    start_date = pd.to_datetime(start_date_str)
    end_date = '2021-12-31'

    end_date = pd.to_datetime(end_date)
    date_range_str = start_date_str + "_" + end_date.strftime("%Y-%m-%d")

    data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)

    weather_data = data_creator.create_data()

    selected_model_params = model_params[model_name]["core"]
    batch_gen_params = model_params[model_name]["batch_gen"]
    trainer_params = model_params[model_name]["trainer"]

    config = {
        "data_params": data_params,
        "experiment_params": experiment_params,
        f"{model_name}_params": model_params[model_name]
    }

    batch_generator = BatchGenerator(weather_data=weather_data,
                                     val_ratio=val_ratio,
                                     test_ratio=test_ratio,
                                     params=batch_gen_params,
                                     normalize_flag=normalize_flag)

    model = model_dispatcher[model_name](device=device, **selected_model_params)
    '''
    print(f"Training {model_name} for the {date_range_str}")
    train(model_name=model_name,
          model=model,
          batch_generator=batch_generator,
          trainer_params=trainer_params,
          date_r=date_range_str,
          config=config,
          device=device)
'''

    print(f"Predicting {model_name} for the {date_range_str}")
    try:
        predict(model_name=model_name, batch_generator=batch_generator, device=device
                ,exp_num=12)

    except Exception as e:
        print(f"Couldnt perform prediction, the exception is {e}")


if __name__ == '__main__':
    run()
