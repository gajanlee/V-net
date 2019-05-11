import keras.backend as K
from params import Params
from keras.losses import binary_crossentropy


def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_probabilities(true_and_pred):
        y_true, y_pred = true_and_pred
        losses = []

        def get_loss_per_passage(true_and_pred):
            y_true, y_pred = true_and_pred
            start_probability = y_pred[0, K.cast(y_true[0], dtype="int32")]
            end_probability = y_pred[1, K.cast(y_true[1], dtype="int32")]
            return K.log(start_probability) + K.log(end_probability)

        passage_loss_sum = K.map_fn(get_loss_per_passage, (y_true, y_pred), dtype="float32")
        return K.mean(passage_loss_sum, axis=0)

    y_true = K.squeeze(y_true, axis=-1)

    batch_probability_sum = K.map_fn(sum_of_log_probabilities, (y_true, y_pred), dtype='float32')
    return -K.mean(batch_probability_sum, axis=0)


"""
    y_truth: (None, Params.max_passage_count, Params.max_passage_len)   1 or 0
    y_pred : ...    probability of word in answer
"""
def content_loss_function(y_truth, y_pred):
    def cross_entropy(truth_and_pred):
        y_truth, y_pred = truth_and_pred
        def cross_entropy_per_passage(truth_and_pred):
            y_truth, y_pred = truth_and_pred
            return -K.mean(K.binary_crossentropy(y_truth, y_pred), axis=-1)

        passage_loss = K.map_fn(cross_entropy_per_passage, (y_truth, y_pred), dtype="float32")
        return K.mean(passage_loss)

    batch_content_loss = K.map_fn(cross_entropy, (y_truth, y_pred), dtype="float32")
    return -K.mean(batch_content_loss)


"""
    y_truth: (None, Params.max_passage_count)   1 or 0 represents if document contains answer
    y_pred : ...
"""
def verify_loss_function(y_truth, y_pred):
    return -K.mean(K.log(K.map_fn(lambda truth, pred: truth*pred, (y_truth, y_pred))))