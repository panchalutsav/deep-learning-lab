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
def evaluate(model, ds_test, ds_info, ckpt_path=False, log_wandb=False):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Restore model to the latest checkpoint
    if ckpt_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
        logging.info(f"Check point restored from {ckpt_path} ")

    confusion_matrix = ConfusionMatrix()
    auc_metric = tf.keras.metrics.AUC(num_thresholds=50)

    y_true_array = []
    y_pred_array = []

    # perform predictions on test set
    for images, labels in ds_test:
        predictions = model(images, training=False)
        test_accuracy.update_state(labels, predictions)
        y_pred = tf.argmax(predictions, axis=1)
        y_pred_array.append(y_pred)
        y_true = tf.squeeze(labels, axis=-1)
        y_true_array.append(y_true)
        confusion_matrix.update_state(y_true, y_pred)
        auc_metric.update_state(y_true, y_pred)
        ub_accuracy, recall, precision, f1_score, sensitivity, specificity, balanced_accuracy = confusion_matrix.get_related_metrics()

        if log_wandb:
            wandb.log(
                {'test_sparse_acc': test_accuracy.result() * 100, 'test_balanced_accuracy': balanced_accuracy * 100, \
                 'test_sensitivity': sensitivity, 'test_specificity': specificity})

    cm_result = confusion_matrix.result()
    auc_result = auc_metric.result()
    ub_accuracy, recall, precision, f1_score, sensitivity, specificity, balanced_accuracy = confusion_matrix.get_related_metrics()
    confusion_matrix.reset_state()
    auc_metric.reset_state()
    sparse_accuracy = test_accuracy.result()

    logging.info(f"\n====Results of Test set evaluation on {model.name} ====")
    logging.info(f"Confusion Matrix: {cm_result.numpy()[0]} {cm_result.numpy()[1]}")
    logging.info("Accuracy(balanced): {:.2f}".format(balanced_accuracy * 100))
    logging.info("Accuracy(Unbalanced): {:.2f}".format(ub_accuracy * 100))
    logging.info("Accuracy(Sparse Categorical) {:.2f}".format(sparse_accuracy * 100))
    logging.info("sensitivity: {:.2f}".format(sensitivity * 100))
    logging.info("specificity: {:.2f}".format(specificity * 100))
    logging.info("recall: {:.2f}".format(recall * 100))
    logging.info("precision: {:.2f}".format(precision * 100))
    logging.info("f1_score: {:.2f}".format(f1_score * 100))
    logging.info("AUC {:.2f}".format(auc_result.numpy()))

    # Get curves
    plot_confusion_mat(cm_result)
    plot_roc_curve(y_true_array, y_pred_array)

    logging.info("----Evaluation completed----")
    return


def plot_roc_curve(true_array, prediction_array):
    """
    Convert the array to 1 dim array first. Necessary for ROC AUC Curve input
    """
    index = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    true_array = np.concatenate([tensor.numpy() for tensor in true_array])
    prediction_array = np.concatenate([tensor.numpy() for tensor in prediction_array])

    fpr, tpr, thresholds = metrics.roc_curve(true_array, prediction_array, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    lw = 2
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.15)
    plt.plot(fpr, tpr, lw=lw, label=f'ROC curve (area = {roc_auc: 0.2f})')

    plt.xlabel('(1 – Specificity) - False Positive Rate')
    plt.ylabel('Sensitivity - True Positive Rate')
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f"ROC_AUC_Curve_{index}.png")


def plot_confusion_mat(cm):
    index = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()
    plt.savefig(f"heatmap_{index}.png")
