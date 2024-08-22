experiment_params = {
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "normalize_flag": True,
    "model": "cru_net",
    "device": 'cuda'
}

data_params = {
    "weather_raw_dir": 'H:\\OSTIA',
    "spatial_range": [[12,16], [115, 120]],
    "weather_freq": 1,
    "downsample_mode": "selective",
    "check_files": False,
    "features": ['analysed_sst'],
    "target_dim": 0,
    "rebuild": True
}

model_params = {

    "cru_net": {
        "batch_gen": {
            "input_dim": 0,
            "output_dim": 0,
            "window_in_len": 10,
            "window_out_len":10,
            "batch_size": 1,
            "shuffle": False,
        },
        "trainer": {
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.0002,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 8
        },
        "core": {
            "selected_dim": 0,
            "in_channels": 10,
            "out_channels": 10
        }
    },

}
