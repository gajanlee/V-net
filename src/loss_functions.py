import tensorflow as tf
import keras.backend as K

def boundary_loss(y_true, y_pred):
    """
        Calculate Boundary Loss
        : y_true, shape of (batch_size, max_passage_count, 2, 1), 
            the shape 2 represents the start and end position, 
            accumulate from 0.
        : y_pred, shape of (batch_size, max_passage_count, 2, max_passage_len),
            the shape of max_passage_len represents the probability of selected
            for start or end probability.
    """
    def boundary_loss_per_passage(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return K.log(start_probability) + K.log(end_probability)

    def boundary_loss_passages(true_and_pred):
        y_true, y_pred = true_and_pred

        y_pred_start = y_pred[:, 0, :]
        y_pred_end = y_pred[:, 1, :]
        passages_losses = K.map_fn(boundary_loss_per_passage, (y_true, y_pred_start, y_pred_end), dtype="float32")
        return K.mean(passages_losses)
    
    y_true = K.squeeze(y_true, axis=-2)
    batch_probability_sum = K.map_fn(boundary_loss_passages, (y_true, y_pred), dtype='float32')
    return -K.mean(batch_probability_sum, axis=0)


def content_loss(y_true, y_pred):
    
    def content_loss_per_passage(true_and_pred):
        y_true, y_pred = true_and_pred
        return K.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true), axis=-1)

    def content_loss_passages(true_and_pred):
        y_true, y_pred = true_and_pred
        passages_loss = K.map_fn(content_loss_per_passage, (y_true, y_pred), dtype="float32")
        return K.mean(passages_loss)
    
    batch_content_loss = K.map_fn(content_loss_passages, (y_true, y_pred), dtype="float32")
    return K.mean(batch_content_loss)


def verify_loss(y_true, y_pred):
    batch_losses = K.sum(y_true * K.log(y_pred), axis=-1)
    #batch_losses = K.expand_dims(batch_losses, axis=-1)
    # batch_losses = K.map_fn(loss_per_batch, (y_true, y_pred), dtype="float32")
    return -K.mean(batch_losses, axis=-1)