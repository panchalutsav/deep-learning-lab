import time

import gin
import tensorflow as tf
import logging
import wandb

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval,
                 learning_rate, ckpt_path=False, log_wandb=False):
       
        # Checkpoint Manager
        self.model = model
        self.ckpt_path = ckpt_path
        self.ckpt = tf.train.Checkpoint(model=self.model)
        if self.ckpt_path:
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=4)
            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                logging.info("Restored from {}".format(self.manager.latest_checkpoint))
            else:
                logging.error("Checkpoint path provided is not valid : {}".format(self.ckpt_path))
        else:
            self.manager = tf.train.CheckpointManager(self.ckpt, run_paths["path_ckpts_train"], max_to_keep=4)
            logging.info(f"Initializing from scratch. Checkpoints stored in {run_paths['path_ckpts_train']}")

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.log_wandb = log_wandb

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            start_time = time.time()
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Time {:.2f}, Loss: {:.2f}, Accuracy: {:.2f}, Validation Loss: {:.2f}, Validation ' \
                           'Accuracy: {:.2f}'
                logging.info(template.format(step,
                                             time.time() - start_time,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # wandb logging: make the flag true from config.gin to start logging
                if self.log_wandb:
                    wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                               'val_acc': self.val_accuracy.result() * 100, 'val_loss': self.val_loss.result(),
                               'step': step})

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                return self.val_accuracy.result().numpy()

@gin.configurable
class TransferTrainer(Trainer):
    def __init__(self, model, base_model, ds_train, ds_val, ds_info, run_paths, 
                 coarse_learning_rate, fine_learning_rate, coarse_num_steps, fine_num_steps, layers_to_tune):
        super().__init__(model=model, ds_train=ds_train, ds_val=ds_val, ds_info=ds_info, run_paths=run_paths,
                        total_steps=1000, 
                        log_interval=gin.query_parameter("Trainer.log_interval"),
                        ckpt_interval=gin.query_parameter("Trainer.ckpt_interval"), 
                        learning_rate = 0.001, ckpt_path=False, log_wandb=gin.query_parameter("Trainer.log_wandb") )
        
        self.base_model = base_model
        self.coarse_learning_rate = coarse_learning_rate
        self.fine_learning_rate = fine_learning_rate
        self.coarse_num_steps = coarse_num_steps
        self.fine_num_steps = fine_num_steps
        self.layers_to_tune = layers_to_tune
        self.fine_training_alone = self.coarse_num_steps <= 0
        self.coarse_training_alone = self.fine_num_steps <= 0


    def train(self):
        logging.info("\n=====================Transfer Training======================")
        logging.info(f"The total number of layers in base model  = {len(self.base_model.layers)}")
        if not self.fine_training_alone:
            yield from self.coarse_training()
            logging.info("\n Training completed")
        if not self.coarse_training_alone:
            yield from self.fine_training()
            logging.info("\n Training completed")


    def coarse_training(self):
        # coarse_training
        logging.info("Starting coarse training of model's classification head")
        self.setup_layers_training_params(is_fine_tuning=False)
        yield from super().train()

    
    def fine_training(self):
        # fine_training
        logging.info("Starting fine tuning of the model")
        self.setup_layers_training_params(is_fine_tuning=True)
        yield from super().train()


    def setup_layers_training_params(self, is_fine_tuning):
        #params
        self.base_model.trainable = False
        self.learning_rate = self.fine_learning_rate if is_fine_tuning else self.coarse_learning_rate
        self.total_steps = self.fine_num_steps if is_fine_tuning else self.coarse_num_steps
        #layers
        if is_fine_tuning:
            self.base_model.trainable = True
            for layer in self.base_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
            for layer in self.base_model.layers[:-self.layers_to_tune]:
                layer.trainable = False
        self.model.summary()
    
        

    

    
    
        


    

    


