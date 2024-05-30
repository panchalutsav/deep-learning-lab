import logging
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import vgg_like, cnn_1, cnn_se
from train import Trainer
from utils import utils_params, utils_misc
from evaluation.eval import evaluate
import os
from train import Trainer
from input_pipeline import tfrecords
from evaluation.eval import evaluate
from absl import app



def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        # run_paths = utils_params.gen_run_folder(','.join(bindings))
        run_paths = utils_params.gen_run_folder("cnn_se")

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        run.name = run_paths['path_model_id'].split(os.sep)[-1]

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))

        # model
        model = cnn_se()
        #model = cnn_1()
        model.summary()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue

        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info("Starting model evaluation after sweep")
        evaluate(model,
                 ds_test,
                 ds_info,
                 run_paths
                 )


sweep_config = {
    'name': 'diabetic-retinopathy-sweep',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [17000, 20000, 25000, 30000, 40000]
        },
        'cnn_se.filters': {
            'values': [(4, 8, 16, 32), (4,8,16,32,32)]
        },
        'cnn_se.kernel_size': {
            'values': [3, 5]
        },
        'cnn_se.strides': {
            'values': [(1, 1, 1, 1, 1, 1)]
        },
        'cnn_se.pool_size': {
            'values': [2, 3]
        },
        'cnn_se.dropout_blocks': {
            'values': [(1, 2)]
        },
        'cnn_se.maxpool_blocks': {
            'values': [(1, 2)]
        },
        'cnn_se.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.4
        }
    },
    'count': 10  # Set the total number of runs
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=50)
