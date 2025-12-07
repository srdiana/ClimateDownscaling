DATA_CONFIG = {
    'data_path': '../../main_phys_downscaling_code/dataset/train_6h',
    'years': [2018, 2019],
    'variables': ['T2M', 'MSL', 'TP', '10u', '10v'],
    'input_scale': 'low',
    'target_scale': 'high', 
    'spatial_ratio': 4,
    'normalize': True
}
MODEL_CONFIG = {
    'unet': {
        'hidden_dim': 64
    },
    'neural_ode': {
        'hidden_dim': 64,
        'ode_hidden_dim': 64,
        'use_topography': True
    },
    'residual_ode': {
        'hidden_dim': 64,
        'num_blocks': 3
    }
}
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'early_stopping_patience': 15
}

SETTINGS = {
    'use_topography': True,
    'experiment_name': 'baseline_experiment'
}