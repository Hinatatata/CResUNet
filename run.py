import pandas as pd
from config import experiment_params, data_params, model_params
from data_creator import DataCreator
from batch_generator import BatchGenerator
from experiment import train, predict
from models.baseline.CResU_Net import CRUNet
import warnings
warnings.filterwarnings("ignore")


def run():

    normalize_flag = experiment_params['normalize_flag']
    model_name = experiment_params['model']
    device = experiment_params['device']
    operation_mode = experiment_params['operation_mode']

    model_dispatcher = {
        'CResU_Net': CRUNet,
    }

    period_keys = ["train", "val", "test"]
    date_ranges = {key: (pd.to_datetime(data_params[f"{key}_period"]["start"]),
                         pd.to_datetime(data_params[f"{key}_period"]["end"]))
                   for key in period_keys}

    date_range_strs = {}
    data_creators = {}
    data_sets = {}

    for key in period_keys:
        start_date, end_date = date_ranges[key]
        date_range_strs[key] = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        data_creators[key] = DataCreator(start_date=start_date, end_date=end_date, **data_params)
        data_sets[key] = data_creators[key].create_data()

    train_date_range_str = date_range_strs["train"]
    val_date_range_str = date_range_strs["val"]
    test_date_range_str = date_range_strs["test"]
    train_data = data_sets["train"]
    val_data = data_sets["val"]
    test_data = data_sets["test"]

    selected_model_params = model_params[model_name]["core"]
    batch_gen_params = model_params[model_name]["batch_gen"]
    trainer_params = model_params[model_name]["trainer"]

    config = {
        "data_params": data_params,
        "experiment_params": experiment_params,
        f"{model_name}_params": model_params[model_name]
    }

    batch_generator = BatchGenerator(train_data=train_data,
                                     val_data=val_data,
                                     test_data=test_data,
                                     params=batch_gen_params,
                                     normalize_flag=normalize_flag)

    model = model_dispatcher[model_name](device=device, **selected_model_params)

    print(f"Training {model_name} for the {train_date_range_str}, val for the {val_date_range_str}")

    train(model_name=model_name,
          model=model,
          batch_generator=batch_generator,
          trainer_params=trainer_params,
          date_r=train_date_range_str,
          config=config,
          device=device)

    print(f"Predicting {model_name} for the {test_date_range_str}")

    try:
        predict(model_name=model_name, batch_generator=batch_generator, device=device
                , exp_num=None)
    except Exception as e:
        print(f"Cant perform prediction, the exception is {e}")


if __name__ == '__main__':
    run()
