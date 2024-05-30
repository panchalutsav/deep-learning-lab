import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self,n_classes ,name="confusion_matrix" ,**kwargs):
        super(ConfusionMatrix, self).__init__(name=name ,**kwargs)
        self.ConfusionMat = self.add_weight("confusion_matrix", (n_classes, n_classes), dtype=tf.int32, initializer="zeros")
        self.n_classes = n_classes

    def update_state(self, y_true, y_pred):
        mtx = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.n_classes)
        self.ConfusionMat.assign_add(mtx)

    def reset_state(self):
        for x in self.variables:
            x.assign(tf.zeros(shape=x.shape, dtype=tf.int32))

    def result(self):
        return self.ConfusionMat

    def get_related_metrics(self):
        """
        Metrics for Multi-Class Classification: An Overview
        Margherita Grandini, Enrico Bagli, Giorgio Visani
        August 14, 2020
        Reference: https://arxiv.org/pdf/2008.05756.pdf

        recall: number of correct predictions out of total number of real samples in that class.
        precision: number of correct predictions out of total number samples predicted to be in that class.
        """
        tp = tf.linalg.diag_part(self.ConfusionMat)
        fn = tf.reduce_sum(self.ConfusionMat, axis=1) - tp
        fp = tf.reduce_sum(self.ConfusionMat, axis=0) - tp

        accuracy = tf.reduce_sum(tp) / tf.reduce_sum(self.ConfusionMat)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) 
        macro_recall = tf.reduce_mean(recall)
        macro_precision = tf.reduce_mean(precision)
        macro_f1_score = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
        balanced_accuracy = tf.reduce_mean(recall)

        return accuracy.numpy(), recall.numpy(), precision.numpy(), macro_f1_score.numpy(), balanced_accuracy.numpy()
