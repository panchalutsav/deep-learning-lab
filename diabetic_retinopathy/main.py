import gin
import logging
from absl import app, flags
import wandb
import sys

from deep_visu.deep_visualise import DeepVisualize
from train import Trainer, TransferTrainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like, cnn_1, cnn_se, transfer_model
from ensemble_learning import EnsembleModel
from input_pipeline import tfrecords

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train  model.')
flags.DEFINE_boolean('eval', False, 'Specify whether to evaluate  model.')
flags.DEFINE_string('model_name', 'cnn_se', 'Choose model to train. Default model cnn')
flags.DEFINE_string('base_model', 'InceptionV3', 'Choose base model to train')
flags.DEFINE_boolean('deep_visu', False, 'perform deep visualization with grad_cam')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str()) 

    if tfrecords.make_tfrecords():
        logging.info("Created TFRecords files")

    # setup wandb
    wandb.init(project='diabetic-retinopathy', name=run_paths['path_model_id'],
            config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(data_dir=gin.query_parameter('make_tfrecords.target_dir'))

    # model
    if FLAGS.model_name == 'transfer_model':
        model  = transfer_model(base_model_name=FLAGS.base_model)
    elif FLAGS.model_name == 'cnn_se':
        model = cnn_se()
    elif FLAGS.model_name == 'cnn_1':
        model = cnn_1()
    elif FLAGS.model_name == "vgg":
        model = vgg_like()
    elif FLAGS.model_name == "ensemble_model":
        ensemble_model = EnsembleModel()
        model = ensemble_model()
    model.summary()

    if FLAGS.train:
        if FLAGS.model_name == 'ensemble_model':
            logging.error("Trying to train ensemble model, ensemble model cannot be trained.\n Load the checkpoints in config.gin to evaluate the ensemble model.")
            sys.exit(1)
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
        logging.info("Starting model training...")
        
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        #trainer = TransferTrainer(model, base_model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue

    if FLAGS.eval:
        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)
        logging.info(f"Starting model evaluation...")
        evaluate(model,
                ds_test,
                ds_info
                )

    if FLAGS.deep_visu:
        deep_visualize = DeepVisualize(model, run_paths, data_dir=gin.query_parameter('make_tfrecords.data_dir'))
        deep_visualize.visualize()


if __name__ == "__main__":
    app.run(main)
