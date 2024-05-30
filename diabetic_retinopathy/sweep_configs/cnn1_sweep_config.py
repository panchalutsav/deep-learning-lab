sweep_config = {
    'name': 'diabetic-retinopathy-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [12000, 15000, 16000, 17000]
        },
        'cnn_1.filters ': {
            'values': [(4, 8, 16, 32), (8,16,32,64,128), (8,16,16,32,64)]
        },
        'cnn_1.kernel_size': {
            'values': [3, 5, 7]
        },
        'cnn_1.strides': {
            'values': [(1, 1, 1, 12)]
        },
        'cnn_1.pool_size': {
            'values': [2, 3]
        },
        'cnn_1.dropout_blocks': {
            'values': [(1, 2), (2, 3)]
        },
        'cnn_1.maxpool_blocks': {
            'values': [(1, 2), (2, 3)]
        },
        'cnn_1.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}