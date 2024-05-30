import time

import gin
import tensorflow as tf
import logging
import wandb

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, class_weights, run_paths, total_steps, log_interval, ckpt_interval,
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
        if class_weights is not None:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        else:
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.class_weights = class_weights
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
    def train_step(self, sequences, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(sequences, training=True)
            loss = self.loss_object(labels, predictions)
            if self.class_weights is not None:
                class_weights = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)
                loss = loss * tf.gather(class_weights, labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, sequences, labels):
        predictions = self.model(sequences, training=False)
        loss = self.loss_object(labels, predictions)
        if self.class_weights is not None:
            class_weights = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)
            loss = loss * tf.gather(class_weights, labels)
    
        self.val_loss(loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (sequences, labels) in enumerate(self.ds_train):

            step = idx + 1
            start_time = time.time()
            self.train_step(sequences, labels)

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_sequences, val_labels in self.ds_val:
                    self.val_step(val_sequences, val_labels)

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

