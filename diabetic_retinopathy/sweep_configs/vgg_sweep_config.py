sweep_config = {
    'name': 'diabetic-retinopathy-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [5000, 7000, 7500, 8000, 10000]
        },
        'Trainer.learning_rate': {
            'values': [0.001, 0.003, 0.005, 0.0001, 0.0003]
        },
        'vgg_like.base_filters': {
            'values': [4, 8, 16]
        },
        'vgg_like.n_blocks': {
            'values': [4,5,6,7]
        },
        'vgg_like.dense_units': {
            'values': [16, 32, 64]
        },
        'vgg_like.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}