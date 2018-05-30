import tensorflow as tf


def masked_softmax_cross_entropy(preds_ll, labels_ll):
    """Softmax cross-entropy loss with masking."""
    lo = 0.0
    ll=[]
    for i in range(len(preds_ll)):
        preds, labels=preds_ll[i], labels_ll[i]
        loss = tf.abs(labels - preds)

        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        # loss *= mask
        lo+=tf.reduce_mean(loss)
        ll.append(tf.reduce_mean(loss))
    return lo,ll


def masked_loss(preds_ll, labels_ll,mask_ll):
    loss = 0.0
    for i in range(len(preds_ll)):
        preds, labels,mask= preds_ll[i], labels_ll[i],mask_ll
        accuracy_all=tf.abs(preds-labels)
        # accuracy_all = tf.where(tf.greater(preds,labels),(preds-labels),(labels-preds))
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy_all *= mask
        loss+=tf.reduce_mean(accuracy_all)
    return loss


def masked_accuracy(preds_ll, labels_ll,mask_ll):
    loss = 0.0
    for i in range(len(preds_ll)):
        preds, labels,mask= preds_ll[i], labels_ll[i],mask_ll[i]
        preds=tf.nn.relu(preds)
        accuracy_all = tf.div(tf.abs(labels - preds), tf.abs(labels))
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy_all *= mask
        loss+=tf.reduce_mean(accuracy_all)
    return loss


