sweep_config = {
    'name': 'diabetic-retinopathy-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [8000, 12000, 15000, 16000, 17000, 20000, 30000, 40000]
        },
        'cnn_se.filters ': {
            'values': [(4, 8, 16, 32), (4,8,16,32,32), (4, 8, 16, 32, 64), (8,16,16,32,64)]
        },
        'cnn_se.kernel_size': {
            'values': [3, 5, 7]
        },
        'cnn_se.strides': {
            'values': [(1, 1, 1, 12)]
        },
        'cnn_se.pool_size': {
            'values': [2, 3]
        },
        'cnn_se.dropout_blocks': {
            'values': [(1, 2), (2, 3)]
        },
        'cnn_se.maxpool_blocks': {
            'values': [(1, 2), (2, 3)]
        },
        'cnn_se.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'cnn_se.batch_norm_blocks' : {
            'values': [(0,), (2,), (0, 1), (0, 2), False]
        }
    }
}