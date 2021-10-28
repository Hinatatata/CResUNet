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

    "convlstm": {
        "batch_gen": {
            "input_dim": 0,
            "output_dim": 0,
            "window_in_len": 10,
            "window_out_len": 10,
            "batch_size": 1,
            "shuffle": False,
        },
        "trainer": {
            "num_epochs": 100,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 6
        },

        "core": {
            "input_size": (80, 100),
            "window_in": 10,
            "window_out": 10,
            "num_layers": 3,
            "encoder_params": {
                "input_dim": 1,
                "hidden_dims": [1, 16, 32],
                "kernel_size": [3, 3, 3],
                "bias": False,
                "peephole_con": False
            },
            "decoder_params": {
                "input_dim": 32,
                "hidden_dims": [32, 16, 1],
                "kernel_size": [3, 3, 3],
                "bias": False,
                "peephole_con": False
            }
        },
    },

    "u_net": {
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
            "weight_decay": 0.00023,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "selected_dim": 0,
            "in_channels": 10,
            "out_channels": 10
        }
    },

    "SmaAt_UNet": {
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
            "weight_decay": 0.00023,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "selected_dim": 0,
            "in_channels": 10,
            "out_channels": 10
        }
    },

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
            "weight_decay": 0.00023,
            "learning_rate": 0.0001,
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "selected_dim": 0,
            "in_channels": 10,
            "out_channels": 10
        }
    },

}
