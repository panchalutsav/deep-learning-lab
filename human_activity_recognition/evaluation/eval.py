import logging

import gin
import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
import wandb
from datetime import datetime

@gin.configurable
def evaluate(model, ds_test, ds_info,activity_labels,ckpt_path=False ,log_wandb=False):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # Restore model to the latest checkpoint
    if ckpt_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
        logging.info(f"Check point restored from {ckpt_path} ")

    confusion_matrix = ConfusionMatrix(n_classes=model.output_shape[1])

    y_true_array = []
    y_pred_array = []

    # perform predictions on test set
    for sequences, labels in ds_test:
        predictions = model(sequences, training=False)
        test_accuracy.update_state(labels, predictions)
        y_pred = tf.argmax(predictions, axis=1)
        y_pred_array.append(y_pred)
        y_true = tf.squeeze(labels, axis=-1)
        y_true_array.append(y_true)
        confusion_matrix.update_state(y_true, y_pred)
        ub_accuracy, recall, precision, macro_f1_score, balanced_accuracy = confusion_matrix.get_related_metrics()

        if log_wandb:
            wandb.log(
                {'test_sparse_acc': test_accuracy.result() * 100, 'test_balanced_accuracy': balanced_accuracy * 100, 'macro_f1_score': macro_f1_score*100})

    cm_result = confusion_matrix.result()
    ub_accuracy, recall, precision, macro_f1_score, balanced_accuracy = confusion_matrix.get_related_metrics()
    confusion_matrix.reset_state()
    sparse_accuracy = test_accuracy.result()
    formatted_recall = ["{:.2f}%".format(value*100) for value in recall]
    fromatted_precision = ["{:.2f}%".format(value*100) for value in precision]

    logging.info(f"\n====Results of Test set evaluation on {model.name} ====")
    logging.info(f"Confusion Matrix:\n{np.array2string(cm_result.numpy(), separator=' ', max_line_width=np.inf)}")
    logging.info("Accuracy(balanced): {:.2f}".format(balanced_accuracy * 100))
    logging.info("Accuracy(Unbalanced): {:.2f}".format(ub_accuracy * 100))
    logging.info("Accuracy(Sparse Categorical) {:.2f}".format(sparse_accuracy * 100))
    logging.info("recall: {}".format(formatted_recall))
    logging.info("precision: {}".format(fromatted_precision))
    logging.info("macro_f1_score: {:.2f}".format(macro_f1_score * 100))

    # Get curves
    plot_confusion_mat(cm_result, activity_labels)
    logging.info("----Evaluation completed----")
    return


def plot_confusion_mat(cm, activity_labels):
    index = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d')
    
    # Set custom xticks and yticks
    plt.xticks(ticks=range(len(activity_labels)), labels=activity_labels, rotation=60 )
    plt.yticks(ticks=range(len(activity_labels)), labels=activity_labels, rotation=60)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()
    plt.savefig(f"heatmap_{index}.png")
