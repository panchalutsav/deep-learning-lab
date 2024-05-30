import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.ConfusionMat = self.add_weight("confusion_matrix", (2, 2), dtype=tf.int32, initializer="zeros")

    def update_state(self, y_true, y_pred):
        mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        self.ConfusionMat.assign_add(mtx)

    def reset_state(self):
        for x in self.variables:
            x.assign(tf.zeros(shape=x.shape, dtype=tf.int32))

    def result(self):
        return self.ConfusionMat

    def get_related_metrics(self):
        """
        WIKI: https://neptune.ai/blog/balanced-accuracy
        """
        tp = self.ConfusionMat[0, 0]
        tn = self.ConfusionMat[1, 1]
        fp = self.ConfusionMat[0, 1]
        fn = self.ConfusionMat[1, 0]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2

        return accuracy.numpy(), recall.numpy(), precision.numpy(), f1_score.numpy(), sensitivity.numpy(), \
            specificity.numpy(), balanced_accuracy.numpy()
