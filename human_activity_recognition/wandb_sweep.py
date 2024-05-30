import logging
import wandb
import gin
import math

from input_pipeline.datasets import load
from models.architectures import model1_LSTM
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
        run_paths = utils_params.gen_run_folder("model1_LSTM")

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        run.name = run_paths['path_model_id'].split(os.sep)[-1]

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info, class_weights = load(name="har", data_dir=gin.query_parameter('make_tfrecords.target_dir'))
        window_length = gin.query_parameter('make_tfrecords.window_length')
        # model
        model = model1_LSTM(window_length=window_length)
        model.summary()

        trainer = Trainer(model, ds_train, ds_val, ds_info, class_weights, run_paths)
        for _ in trainer.train():
            continue

        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info("Starting model evaluation after sweep")
        evaluate(model, ds_test, ds_info)


# method: 'bayes', 'grid', 'random'
sweep_config = {
    'name': 'hapt-LSTM-sweep1',
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps': {
            'values': [2500, 3000, 3500, 4000]
        },
        'model1_LSTM.num_lstm': {
            'values': [2, 3, 4, 5]
        },
        'model1_LSTM.lstm_cells': {
            'values': [32, 64,128,256]
        },
        'model1_LSTM.dense_units': {
            'values': [8,16,32,64]
        },
        'model1_LSTM.dropout_rate': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.4
        }
    },
    'count': 10  # Set the total number of runs
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=50)
